import json

def load_json(json_file):
    try:
        with open(json_file) as f:
            params, training_stats = json.load(f)
        training_curve = training_stats['training_curve']
        _, train_loss, test_loss = training_curve[-1]
        _, halfway_train_loss, _ = training_curve[len(training_curve)/2]
        return {'params' : params,
                'varied_params' : params['varied_params'],
                'train_loss' : train_loss,
                'test_loss'  : test_loss,
                'training_curve' : training_curve,
                'halfway_train_loss' : halfway_train_loss}
    except ValueError:
        return None

def get_losses(loss_name):
    return map(lambda x : x[loss_name], jobs_data)

def get_hypers(hyper_name):
    return map(lambda x : x['varied_params'][hyper_name], jobs_data)


def get_jobs_data(data_file_names):
    return filter(bool, map(load_json, data_file_names))
