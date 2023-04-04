import argparse
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from data import DataFetch, Data, GraphData
from torch_geometric.loader import DataLoader
from model import Baseline, BaselineLongitudinal, GNN, GNNLongitudinal, GNNExplainabilityLSTM
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import EarlyStopping, get_classification_threshold_auc, get_classification_threshold_precision_recall, WeightedBCELoss
import explainability
import json


def get_model_output(model, data_batch, params):
    """Handles the different data and model input formats for the models
    """
    if params['model_type'] == 'baseline' and params['include_longitudinal']:
        x_static, x_longitudinal, y = data_batch[0][0].to(params['device']), data_batch[1][0].to(params['device']), data_batch[2][0].unsqueeze(1).to(params['device'])
        output = model(x_static, x_longitudinal)
        model_output = {'output':output}
    elif params['model_type'] == 'baseline' and not params['include_longitudinal']:
        x_static, y = data_batch[0][0].to(params['device']), data_batch[1][0].unsqueeze(1).to(params['device'])
        output = model(x_static)
        model_output = {'output':output}
    elif params['model_type'] == 'graph' and params['include_longitudinal']:
        x_static_node, x_static_graph, x_longitudinal_node, x_longitudinal_graph, y, edge_index, edge_weight, batch, target_index = data_batch.patient_x_static.to(params['device']), data_batch.x.to(params['device']), data_batch.patient_x_longitudinal.to(params['device']), data_batch.x_longitudinal.to(params['device']), data_batch.y.unsqueeze(1).to(params['device']), data_batch.edge_index.to(params['device']), data_batch.edge_attr.to(params['device']), data_batch.batch.to(params['device']), data_batch.target_index.to(params['device'])
        output, patient_output, family_output, lstm_output = model(x_static_node, x_static_graph, x_longitudinal_node, x_longitudinal_graph, edge_index, edge_weight, batch, target_index)
        model_output = {'output':output, 'patient_output':patient_output, 'family_output':family_output, 'lstm_output':lstm_output}
    elif params['model_type'] == 'graph' and not params['include_longitudinal']:
        x_static_node, x_static_graph, y, edge_index, edge_weight, batch, target_index = data_batch.patient_x_static.to(params['device']), data_batch.x.to(params['device']), data_batch.y.unsqueeze(1).to(params['device']), data_batch.edge_index.to(params['device']), data_batch.edge_attr.to(params['device']), data_batch.batch.to(params['device']), data_batch.target_index.to(params['device'])
        output, patient_output, family_output = model(x_static_node, x_static_graph, edge_index, edge_weight, batch, target_index)
        model_output = {'output':output, 'patient_output':patient_output, 'family_output':family_output} 
    elif params['model_type'] == 'explainability' and params['include_longitudinal']:
        x_static_node, x_static_graph, x_longitudinal_node, x_longitudinal_graph, y, edge_index, edge_weight, batch, target_index = data_batch.patient_x_static.to(params['device']), data_batch.x.to(params['device']), data_batch.patient_x_longitudinal.to(params['device']), data_batch.x_longitudinal.to(params['device']), data_batch.y.unsqueeze(1).to(params['device']), data_batch.edge_index.to(params['device']), data_batch.edge_attr.to(params['device']), data_batch.batch.to(params['device']), data_batch.target_index.to(params['device'])
        x_longitudinal_graph_flat = torch.reshape(x_longitudinal_graph, (x_longitudinal_graph.shape[0],-1))
        output, patient_output, family_output, lstm_output = model(x_longitudinal_graph_flat, edge_index, edge_weight=edge_weight, x_static_node=x_static_node, x_longitudinal_node=x_longitudinal_node, x_static_graph=x_static_graph, batch=batch, target_node=target_index)
        model_output = {'output':output, 'patient_output':patient_output, 'family_output':family_output, 'lstm_output':lstm_output}

    return model_output, y


def train_model(model, train_loader, validate_loader, params):

    model.to(params['device'])
    model_path = '{}/checkpoint_{}.pt'.format(params['outpath'], params['outname'])
    early_stopping = EarlyStopping(patience=params['patience'], path=model_path)

    if params['loss']=='bce_weighted_single' or params['loss']=='bce_weighted_sum':
        print("Using BCE weighted loss")
        train_criterion = WeightedBCELoss(params['num_samples_train_dataset'], params['num_samples_train_minority_class'], params['num_samples_train_majority_class'], params['device'])
        valid_criterion = WeightedBCELoss(params['num_samples_valid_dataset'], params['num_samples_valid_minority_class'], params['num_samples_valid_majority_class'], params['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_losses = []
    valid_losses = []
    separate_loss_terms = {'NN_train':[], 'target_train':[], 'family_train':[], 'lstm_train':[], 'NN_valid':[], 'target_valid':[], 'family_valid':[], 'lstm_valid':[]}
    
    # store for calculating classification threshold on last epoch
    valid_output = np.array([])
    valid_y = np.array([])

    for epoch in range(params['max_epochs']):
        # update model on train set
        model.train()
        epoch_train_loss = []
        separate_loss_terms_epoch = {'NN_train':[], 'target_train':[], 'family_train':[], 'lstm_train':[], 'NN_valid':[], 'target_valid':[], 'family_valid':[], 'lstm_valid':[]}
        for train_batch in tqdm(train_loader, total=params['num_batches_train']):
            output, y = get_model_output(model, train_batch, params)

            if params['loss']=='bce_weighted_sum':
                # combined loss that considers the additive effect of patient and family effects
                loss_term_NN = params['gamma'] * train_criterion(output['output'], y) 
                loss_term_target = params['alpha'] * train_criterion(output['patient_output'], y)
                loss_term_family = params['beta'] * train_criterion(output['family_output'], y)
                separate_loss_terms_epoch['NN_train'].append(loss_term_NN.item()) 
                separate_loss_terms_epoch['target_train'].append(loss_term_target.item())
                separate_loss_terms_epoch['family_train'].append(loss_term_family.item())
                if params['include_longitudinal'] and (params['model_type'] == 'graph' or params['model_type'] == 'explainability'):
                    loss_term_LSTM = params['delta'] * train_criterion(output['lstm_output'], y) 
                    separate_loss_terms_epoch['lstm_train'].append(loss_term_LSTM.item())
                    loss = loss_term_NN + loss_term_target + loss_term_family + loss_term_LSTM
                else:
                    loss = loss_term_NN + loss_term_target + loss_term_family
            else:
                loss = train_criterion(output['output'], y) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        
        # evaluate on validation set
        model.eval()
        epoch_valid_loss = []
        for validate_batch in tqdm(validate_loader, total=params['num_batches_validate']):
            output, y = get_model_output(model, validate_batch, params)
            valid_output = np.concatenate((valid_output, output['output'].reshape(-1).detach().cpu().numpy()))
            valid_y = np.concatenate((valid_y, y.reshape(-1).detach().cpu().numpy()))

            if params['loss']=='bce_weighted_sum':
                # combined loss that considers the additive effect of patient and family effects
                loss_term_NN = params['gamma'] * valid_criterion(output['output'], y) 
                loss_term_target = params['alpha'] * valid_criterion(output['patient_output'], y)
                loss_term_family = params['beta'] * valid_criterion(output['family_output'], y)
                separate_loss_terms_epoch['NN_valid'].append(loss_term_NN.item())
                separate_loss_terms_epoch['target_valid'].append(loss_term_target.item())
                separate_loss_terms_epoch['family_valid'].append(loss_term_family.item())
                if params['include_longitudinal'] and (params['model_type'] == 'graph' or params['model_type'] == 'explainability'):
                    loss_term_LSTM = params['delta'] * valid_criterion(output['lstm_output'], y) 
                    separate_loss_terms_epoch['lstm_valid'].append(loss_term_LSTM.item())
                    loss = loss_term_NN + loss_term_target + loss_term_family + loss_term_LSTM
                else:
                    loss = loss_term_NN + loss_term_target + loss_term_family
            else:
                loss = valid_criterion(output['output'], y) 

            epoch_valid_loss.append(loss.item())

        early_stopping(np.mean(epoch_valid_loss), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        train_losses.append(np.mean(epoch_train_loss))
        valid_losses.append(np.mean(epoch_valid_loss))
        for term_name in separate_loss_terms:
            separate_loss_terms[term_name].append(np.mean(separate_loss_terms_epoch[term_name]))
        print("epoch {}\ttrain loss : {}\tvalidate loss : {}".format(epoch, np.mean(epoch_train_loss), np.mean(epoch_valid_loss)))

    # load the checkpoint with the best model
    model.load_state_dict(torch.load(model_path))

    # use last values from validation set
    if params['threshold_opt'] == 'auc':
        threshold = get_classification_threshold_auc(valid_output, valid_y)
    elif params['threshold_opt'] == 'precision_recall':
        threshold = get_classification_threshold_precision_recall(valid_output, valid_y)

    plot_losses(train_losses, valid_losses, '{}/{}'.format(params['outpath'], params['outname']))
    
    if params['loss']=='bce_weighted_sum':
        if params['include_longitudinal'] and (params['model_type'] == 'graph' or params['model_type'] == 'explainability'):
            plot_separate_losses(separate_loss_terms['NN_train'], separate_loss_terms['target_train'], separate_loss_terms['family_train'], '{}/{}_train'.format(params['outpath'], params['outname']), lstm_losses=separate_loss_terms['lstm_train'])
            plot_separate_losses(separate_loss_terms['NN_valid'], separate_loss_terms['target_valid'], separate_loss_terms['family_valid'], '{}/{}_validate'.format(params['outpath'], params['outname']), lstm_losses=separate_loss_terms['lstm_valid'])
        else:
            plot_separate_losses(separate_loss_terms['NN_train'], separate_loss_terms['target_train'], separate_loss_terms['family_train'], '{}/{}_train'.format(params['outpath'], params['outname']))
            plot_separate_losses(separate_loss_terms['NN_valid'], separate_loss_terms['target_valid'], separate_loss_terms['family_valid'], '{}/{}_validate'.format(params['outpath'], params['outname']))

    return model, threshold


def plot_losses(train_losses, valid_losses, outprefix):
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validate')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}_train_loss.png'.format(outprefix))
    plt.clf()


def plot_separate_losses(network_losses, target_losses, family_losses, outprefix, lstm_losses=None):
    plt.plot(network_losses, label='Network')
    plt.plot(target_losses, label='Target')
    plt.plot(family_losses, label='Family')
    if lstm_losses is not None:
        plt.plot(lstm_losses, label='LSTM')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}_separate_loss.png'.format(outprefix))
    plt.clf()


def brier_skill_score(actual_y, predicted_prob_y):
    e = sum(actual_y) / len(actual_y)
    bs_ref = sum((e-actual_y)**2) / len(actual_y)
    bs = sum((predicted_prob_y-actual_y)**2) / len(actual_y)
    bss = 1 - bs / bs_ref
    return bss


def calculate_metrics(actual_y, predicted_y, predicted_prob_y):
    auc_roc = metrics.roc_auc_score(actual_y, predicted_prob_y)
    precision, recall, thresholds = metrics.precision_recall_curve(actual_y, predicted_prob_y)
    auc_prc = metrics.auc(recall, precision)
    mcc = matthews_corrcoef(actual_y, predicted_y)
    tn, fp, fn, tp = confusion_matrix(actual_y, predicted_y).ravel()
    ts = tp / (tp + fn + fp)
    recall = tp / (tp + fn)
    f1 = (2*tp) / (2*tp + fp + fn)
    bss = brier_skill_score(actual_y, predicted_prob_y)
    
    metric_results = {'metric_auc_roc':auc_roc, # typically reported, but can be biased for imbalanced classes
               'metric_auc_prc':auc_prc, # better suited for imbalanced classes
               'metric_f1':f1, # also should be better suited for imbalanced classes
               'metric_recall':recall, # important for medical studies, to reduce misses of positive instances
               'metric_mcc':mcc, # correlation that is suitable for imbalanced classes
               'metric_ts':ts, # suited for rare events, penalizing misclassification as the rare event (fp)
               'metric_bss':bss, # brier skill score, where higher score corresponds better calibration of predicted probabilities
               'true_negatives':tn, 
               'false_positives':fp, 
               'false_negatives':fn, 
               'true_positives':tp}
    
    return metric_results


def enable_dropout(model):
    """
    Function to enable the dropout layers during test-time -
    this is needed to get uncertainty estimates with Monte Carlo dropout techniques
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def test_model(model, test_loader, threshold, params, embeddings=False):
    num_samples = 3 # number of MC samples
    if embeddings: num_samples = 1
    test_output = [np.array([]) for _ in range(num_samples)]
    test_y = [np.array([]) for _ in range(num_samples)]

    representations = pd.DataFrame()

    model.eval()
    enable_dropout(model)
    for sample in range(num_samples):
        counter = 0
        for test_batch in tqdm(test_loader, total=params['num_batches_test']):
            output, y = get_model_output(model, test_batch, params)
            if embeddings:
                # get GNN representations - only tested for longitudinal GNN model
                tmp = pd.DataFrame(activation['combined_conv2'].detach().cpu())
                tmp['node_index'] = test_batch.batch + params['batchsize']*counter
                representations = pd.concat([representations, tmp])
            test_output[sample] = np.concatenate((test_output[sample], output['output'].reshape(-1).detach().cpu().numpy()))
            test_y[sample] = np.concatenate((test_y[sample], y.reshape(-1).detach().cpu().numpy()))
            counter += 1

    if embeddings:
        representations.to_csv('{}/{}_embeddings.csv'.format(params['outpath'], params['outname']), index=None)

    # report standard error for uncertainty
    test_output_se = np.array(test_output).std(axis=0) / np.sqrt(num_samples)

    # take average over all samples to get expected value
    test_output = np.array(test_output).mean(axis=0)
    test_y = np.array(test_y).mean(axis=0)

    results = pd.DataFrame({'actual':test_y, 'pred_raw':test_output, 'pred_raw_se':test_output_se})
    results['pred_binary'] = (results['pred_raw']>threshold).astype(int)
    if embeddings:
        metric_results = None
    else:
        metric_results = calculate_metrics(results['actual'], results['pred_binary'], results['pred_raw'])

    return results, metric_results


def get_model(params):
    if params['model_type'] == 'baseline' and params['include_longitudinal']:
        print("Using baseline model with longitudinal data")
        model = BaselineLongitudinal(params['num_features_static'], params['num_features_longitudinal'], params['main_hidden_dim'], params['lstm_hidden_dim'], params['dropout_rate'])
    elif params['model_type'] == 'baseline' and not params['include_longitudinal']:
        print("Using baseline model")
        model = Baseline(params['num_features_static'], params['main_hidden_dim'], params['dropout_rate'])
    elif params['model_type'] == 'graph' and params['include_longitudinal']:
        print("Using graph model with longitudinal data")
        model = GNNLongitudinal(params['num_features_static'], params['num_features_alt_static'], params['num_features_longitudinal'], params['main_hidden_dim'], params['lstm_hidden_dim'], params['gnn_layer'], params['pooling_method'], params['dropout_rate'], params['ratio'])
    elif params['model_type'] == 'graph' and not params['include_longitudinal']:
        print("Using graph model")
        model = GNN(params['num_features_static'], params['num_features_alt_static'], params['main_hidden_dim'], params['gnn_layer'], params['pooling_method'], params['dropout_rate'], params['ratio'])
    elif params['model_type'] == 'explainability' and params['include_longitudinal']:
        print("Using graph model for explainability with longitudinal data")
        model = GNNExplainabilityLSTM(params['num_features_static'], params['num_features_alt_static'], params['num_features_longitudinal'], params['main_hidden_dim'], params['lstm_hidden_dim'], params['gnn_layer'], params['pooling_method'], params['dropout_rate'])

    return model


def get_data_and_loader(patient_list, fetch_data, params, shuffle=True):
    """
    Parameters:
    patient_list: list of patients (target samples) to load data for
    fetch_data: the data object
    params: dictionary of other parameters
    shuffle: samples in random order if true
    """
    if params['model_type'] == 'baseline':
        dataset = Data(patient_list, fetch_data)
    elif params['model_type'] in ['graph', 'graph_no_target', 'explainability']:
        dataset = GraphData(patient_list, fetch_data)

    if shuffle:
        sample_order = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sample_order = torch.utils.data.sampler.SequentialSampler(dataset)

    sampler = torch.utils.data.sampler.BatchSampler(
        sample_order,
        batch_size=params['batchsize'],
        drop_last=False)

    loader = DataLoader(dataset, sampler=sampler, num_workers=params['num_workers'])
    return dataset, loader


activation = {}
def get_activation(name):
    """Used to get representations learned from intermediate layers
    """
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # the following should be updated for each experiment
    parser.add_argument('--featfile', type=str, help='filepath for featfile csv')
    parser.add_argument('--model_type', type=str, help='one of baseline, graph, graph_no_target or explainability')
    parser.add_argument('--experiment', type=str, help='a unique name for the experiment used in output file prefixes')
    parser.add_argument('--batchsize', type=int, help='batchsize for training, recommend 500 for baselines and 250 for graphs', default=500)
    
    # the following can optionally be configured for each experiment
    parser.add_argument('--alt_featfile', type=str, help='filepath for alternative featfile csv (required for graph model_type, for the target patient)', default=None)
    parser.add_argument('--outpath', type=str, help='directory for results output', default='results')
    parser.add_argument('--sqlpath', type=str, help='filepath for sql db', default='long.db')
    parser.add_argument('--statfile', type=str, help='filepath for statfile csv', default='training_data/statfile.csv')
    parser.add_argument('--maskfile', type=str, help='filepath for maskfile csv', default='training_data/maskfile.csv')
    parser.add_argument('--edgefile', type=str, help='filepath for edgefile csv', default='training_data/edgefile_tree_direct.csv')
    parser.add_argument('--gnn_layer', type=str, help='type of gnn layer to use: gcn, graphconv, gat', default='graphconv')
    parser.add_argument('--pooling_method', type=str, help='type of gnn pooling method to use: target, sum, mean, topkpool_sum, topkpool_mean, sagpool_sum, sagpool_mean', default='target')
    parser.add_argument('--obs_window_start', type=int, help='start year of longitudinal observation window', default=1990)
    parser.add_argument('--obs_window_end', type=int, help='end year of longitudinal observation window', default=2010)
    parser.add_argument('--num_workers', type=int, help='number of workers for data loaders', default=6)
    parser.add_argument('--max_epochs', type=int, help='maximum number of training epochs if early stopping criteria not met', default=100)
    parser.add_argument('--patience', type=int, help='how many epochs to wait for early stopping after last time validation loss improved', default=8)
    parser.add_argument('--learning_rate', type=float, help='learning rate for model training', default=0.001)
    parser.add_argument('--main_hidden_dim', type=int, help='number of hidden dimensions in (non-LSTM) neural network layers', default=20)
    parser.add_argument('--lstm_hidden_dim', type=int, help='number of hidden dimensions in LSTM neural network layers', default=20)
    parser.add_argument('--loss', type=str, help='which loss function to use: bce_weighted_single, bce_weighted_sum', default='bce_weighted_single')
    parser.add_argument('--gamma', type=float, help='weight parameter on the overall NN loss (required for bce_weighted_sum loss)', default=1)
    parser.add_argument('--alpha', type=float, help='weight parameter on the target term of the loss (required for bce_weighted_sum loss)', default=1)
    parser.add_argument('--beta', type=float, help='weight parameter on the family term of the loss (required for bce_weighted_sum loss)', default=1)
    parser.add_argument('--delta', type=float, help='weight parameter on the lstm term of the loss (required for bce_weighted_sum loss, longitudinal models only)', default=1)
    parser.add_argument('--dropout_rate', type=float, help='the dropout rate in the neural networks', default=0.5)
    parser.add_argument('--threshold_opt', type=str, help='what metric to optimize when determining the classification threshold (either auc or precision_recall)', default='precision_recall')
    parser.add_argument('--ratio', type=float, help='the graph pooling ratio for node reduction methods, determining portion of nodes to retain', default=0.5)
    
    # extra parameters used for experiments presented in paper - in general these can be ignored
    parser.add_argument('--local_test', action='store_true', help='local testing flag for longitudinal models with simulated data')
    parser.add_argument('--num_positive_samples', type=int, help='number of case samples from test set used in explainability analysis', default=5000)
    parser.add_argument('--explainability_mode', action='store_true', help='explainability flag for running the post-training analysis')
    parser.add_argument('--embeddings_mode', action='store_true', help='extract the representations learned by the GNN')
    parser.add_argument('--explainer_input', type=str, help='optional explainability input file')
    parser.add_argument('--device', type=str, help='specific device to use, e.g. cuda:1, if not given detects gpu or cpu automatically', default='na')

    args = vars(parser.parse_args())

    sqlpath = args['sqlpath']
    filepaths = {'maskfile':args['maskfile'],
                'featfile':args['featfile'],
                'alt_featfile':args['alt_featfile'],
                'statfile':args['statfile'], 
                'edgefile':args['edgefile']}
    params = {'model_type':args['model_type'],
            'gnn_layer':args['gnn_layer'],
            'pooling_method':args['pooling_method'],
            'outpath':args['outpath'],
            'outname':args['experiment'],
            'obs_window_start':args['obs_window_start'],
            'obs_window_end':args['obs_window_end'], 
            'batchsize':args['batchsize'], 
            'num_workers':args['num_workers'],
            'max_epochs':args['max_epochs'],
            'patience':args['patience'],
            'learning_rate':args['learning_rate'],
            'main_hidden_dim':args['main_hidden_dim'],
            'lstm_hidden_dim':args['lstm_hidden_dim'],
            'loss':args['loss'], 
            'gamma':args['gamma'], 
            'alpha':args['alpha'], 
            'beta':args['beta'], 
            'delta':args['delta'],
            'dropout_rate':args['dropout_rate'], 
            'threshold_opt':args['threshold_opt'], 
            'ratio':args['ratio'], 
            'local_test':args['local_test'], 
            'explainability_mode':args['explainability_mode'], 
            'embeddings_mode':args['embeddings_mode'], 
            'explainer_input':args['explainer_input'],
            'device_specification':args['device'],
            'num_positive_samples':args['num_positive_samples']}
    
    if params['device_specification'] != 'na':
        params['device'] = torch.device(params['device_specification'])
    else:
        params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using {} device".format(params['device']))

    fetch_data = DataFetch(filepaths['maskfile'], filepaths['featfile'], filepaths['statfile'], filepaths['edgefile'], sqlpath, params, alt_featfile=filepaths['alt_featfile'], local=params['local_test'])
    
    train_patient_list = fetch_data.train_patient_list
    params['num_batches_train'] = int(np.ceil(len(train_patient_list)/params['batchsize']))
    params['num_samples_train_dataset'] = len(fetch_data.train_patient_list)
    params['num_samples_train_minority_class'] = fetch_data.num_samples_train_minority_class
    params['num_samples_train_majority_class'] = fetch_data.num_samples_train_majority_class
    validate_patient_list = fetch_data.validate_patient_list
    params['num_batches_validate'] = int(np.ceil(len(validate_patient_list)/params['batchsize']))
    params['num_samples_valid_dataset'] = len(fetch_data.validate_patient_list)
    params['num_samples_valid_minority_class'] = fetch_data.num_samples_valid_minority_class
    params['num_samples_valid_majority_class'] = fetch_data.num_samples_valid_majority_class
    test_patient_list = fetch_data.test_patient_list
    params['num_batches_test'] = int(np.ceil(len(test_patient_list)/params['batchsize']))

    train_dataset, train_loader = get_data_and_loader(train_patient_list, fetch_data, params, shuffle=True)
    validate_dataset, validate_loader = get_data_and_loader(validate_patient_list, fetch_data, params, shuffle=True)
    test_dataset, test_loader = get_data_and_loader(test_patient_list, fetch_data, params, shuffle=False)
    params['include_longitudinal'] = train_dataset.include_longitudinal
    params['num_features_static'] = len(fetch_data.static_features)
    if params['model_type'] in ['graph', 'graph_no_target', 'explainability']: params['num_features_alt_static'] = len(fetch_data.alt_static_features)
    params['num_features_longitudinal'] = len(fetch_data.longitudinal_features)

    model = get_model(params)
    model_path = '{}/{}_model.pth'.format(params['outpath'], params['outname'])
    results_path = '{}/{}_results.csv'.format(params['outpath'], params['outname'])
    stats_path = '{}/{}_stats.csv'.format(params['outpath'], params['outname'])
    

    if params['explainability_mode']:
        results = pd.read_csv(results_path)
        # select graphs to explain
        samples = explainability.sampling(results, num_positive_samples=params['num_positive_samples'], uncertainty_rate=0.9)
        exp_patient_list = test_patient_list[samples]
        # load one graph at a time
        params['batchsize'] = 1
        exp_dataset, exp_loader = get_data_and_loader(exp_patient_list, fetch_data, params, shuffle=False)

        del fetch_data # free up memory no longer needed
        del train_dataset
        del validate_dataset
        del test_dataset

        print("Loading model")
        model.load_state_dict(torch.load(model_path))
        model.to(params['device'])
        torch.backends.cudnn.enabled = False
        explainability.gnn_explainer(model, exp_loader, exp_patient_list, params)
    
    elif params['embeddings_mode']:
        # use same samples used for explainability
        exp_data = pd.read_csv(params['explainer_input'])
        exp_patient_list = torch.tensor([int(e) for e in list(exp_data['target_id'].unique())])
        exp_dataset, exp_loader = get_data_and_loader(exp_patient_list, fetch_data, params, shuffle=False)
        params['num_batches_test'] = int(np.ceil(len(exp_patient_list)/params['batchsize']))
        
        del fetch_data # free up memory no longer needed
        del train_dataset
        del validate_dataset
        del test_dataset
        del exp_data

        print("Loading model")
        model.load_state_dict(torch.load(model_path))
        model.combined_conv2.register_forward_hook(get_activation('combined_conv2'))
        model.to(params['device'])

        stats = pd.read_csv(stats_path)
        stats_dict = dict(zip(stats['name'],stats['value']))
        threshold = float(stats_dict['threshold'])
        results, metric_results = test_model(model, exp_loader, threshold, params, embeddings=True)
        
    else:
        # normal training model
        del fetch_data # free up memory no longer needed
        del train_dataset
        del validate_dataset
        del test_dataset

        # model training
        start_time_train = time.time()
        model, threshold = train_model(model, train_loader, validate_loader, params)
        end_time_train = time.time()
        torch.save(model.state_dict(), model_path)
        params['threshold'] = threshold
        params['training_time'] = end_time_train - start_time_train

        # model testing
        results, metric_results = test_model(model, test_loader, threshold, params)
        results.to_csv(results_path, index=None)
        params.update(metric_results)
        stats = pd.DataFrame({'name':list(params.keys()), 'value':list(params.values())})
        stats.to_csv(stats_path, index=None)