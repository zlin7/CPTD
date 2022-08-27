import os
import torch
import numpy as np

class _PI_Constructor(torch.nn.Module):
    def __init__(self, base_model_path=None, **kwargs):
        super(_PI_Constructor, self).__init__()
        self.base_model_path = base_model_path
        assert os.path.isfile(self.base_model_path)
        self.base_model = torch.load(base_model_path, map_location=kwargs.get('device'))
        for param in self.base_model.parameters(): param.requires_grad = False

        self.kwargs = kwargs

        #some optional stuff
        self._update_cal_loc = 0 #if we want to update the calibration residuals in an online fashion


    def fit(self):
        raise NotImplementedError()

    def calibrate(self, calibration_dataset: torch.utils.data.Dataset, batch_size=32, device=None):
        self.base_model.eval()
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
        preds, ys = [], []
        with torch.no_grad():
            for calibration_example in calibration_loader:
                calibration_example = [_.to(device) for _ in calibration_example]
                sequences, targets, lengths_input, lengths_target = calibration_dataset._sep_data(calibration_example)
                out = self.base_model(sequences)
                preds.append(out)
                ys.append(targets)
        self.calibration_preds = torch.nn.Parameter(torch.cat(preds).float(), requires_grad=False)
        self.calibration_truths = torch.nn.Parameter(torch.cat(ys).float(), requires_grad=False)
        return

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        #The most common nonconformity score
        return (cal_pred - cal_y).abs(), (test_pred - test_y).abs()

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha, **kwargs):
        raise NotImplementedError()

    def predict(self, x, y, alpha=0.05, **kwargs):
        raise NotImplementedError()

class CFRNN(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(CFRNN, self).__init__(base_model_path, **kwargs)

    def predict(self, x, y, alpha=0.05, state=None, gamma=0.05, update_cal=False, **kwargs):
        pred = self.base_model(x, state)
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"
        ret = []
        for b in range(y.shape[0]):
            calibration_scores, scores = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths,
                                                                       pred[b], y[b])
            res = scores_to_intervals(pred[b, :, 0], y[b, :, 0], calibration_scores[:, :, 0], 1, alpha=alpha)
            ret.append(res.T.unsqueeze(-1))
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = pred[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)
        ret = torch.stack(ret)
        return pred, ret

class CPTD_R(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(CPTD_R, self).__init__(base_model_path, **kwargs)

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y, beta=1):
        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T

        base_scores = (pred - y).abs()
        base_rank = torch.argsort(torch.argsort(base_scores, 1),1) / base_scores.shape[1] #get the ranking

        _base_ratios = (base_scores / base_scores.median(1, keepdim=True)[0]).cumsum(0) / torch.ones_like(y).cumsum(0) #$nr$ in the paper
        _sorted_base_ratios = torch.sort(_base_ratios)[0]

        scores = torch.zeros_like(y)
        scores[0] = base_scores[0]

        multiplier = torch.ones_like(y)
        q_post = torch.zeros_like(y)
        q_post[0] = torch.ones_like(y[0]) * 0.5
        for t in range(1, len(scores)):
            q_post_t = ((beta + t - 1) * q_post[t-1] + 1 * base_rank[t-1]) / (beta + t)
            multiplier[t] = torch.quantile(_sorted_base_ratios[t - 1], q_post_t)
            scores[t] = base_scores[t] / multiplier[t]
            q_post[t] = q_post_t
        scores = scores.T.unsqueeze(2)
        return scores[:-1], scores[-1], multiplier[:, -1] #(cal_size, seq_len, 1), (seq_len, 1)


    def predict(self, x, y, alpha=0.1, state=None, update_cal=False, **kwargs):
        pred = self.base_model(x, state)

        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"

        ret = []
        for b in range(y.shape[0]):
            calibration_scores, scores, multipliers = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths, pred[b], y[b])
            res = scores_to_intervals(pred[b, :, 0], y[b, :, 0], calibration_scores[:, :, 0], multipliers, alpha=alpha)
            ret.append(res.T.unsqueeze(-1))
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = pred[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)

        ret = torch.stack(ret)
        return pred, ret

class CPTD_M(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(CPTD_M, self).__init__(base_model_path, **kwargs)


    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y, beta=1):
        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        base_scores = (pred - y).abs()
        cummeans = base_scores.cumsum(0) / torch.ones_like(base_scores, device=base_scores.device).cumsum(0)

        multiplier = torch.ones_like(y)
        multiplier[1:] = cummeans[:-1]
        scores = base_scores / multiplier
        scores = scores.T.unsqueeze(2)
        return scores[:-1], scores[-1], multiplier[:, -1] #(cal_size, seq_len, 1), (seq_len, 1)


    def predict(self, x, y, alpha=0.1, state=None, update_cal=False, **kwargs):
        pred = self.base_model(x, state)
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"

        ret = []
        for b in range(y.shape[0]):
            calibration_scores, scores, multipliers = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths, pred[b], y[b])
            res = scores_to_intervals(pred[b, :, 0], y[b, :, 0], calibration_scores[:, :, 0], multipliers, alpha=alpha)
            ret.append(res.T.unsqueeze(-1))
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = pred[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)

        ret = torch.stack(ret)
        return pred, ret


def scores_to_intervals(pred, Y, cal_scores, scales, alpha=0.1):
    # mapping scores back to lower and upper bound.
    assert len(pred.shape) == len(Y.shape) == 1
    N, L = cal_scores.shape
    device = pred.device
    qs = torch.concat([torch.sort(cal_scores, 0)[0], torch.ones([1, L], device=device) * torch.inf], 0)
    qloc = max(0, min(int(np.ceil((1-alpha) * N)), N))
    w_ts = qs[qloc, :]
    return torch.stack([pred - w_ts * scales, pred + w_ts * scales], 1)
