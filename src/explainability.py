"""
Generates explanations for a trained model by conducting a post-training analysis
"""

import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm import tqdm
import numpy as np
import torch


def sampling(results, num_positive_samples, uncertainty_rate=0.8):
    """Determine which samples to generate explanations for by identifying a balanced set
    of low risk and high risk patients which the model predicted correctly with high confidence

    Parameters:
    results - dataframe of results on test dataset
    num_positive_samples - how many samples from the minority class to include (= num_negative_samples)
    uncertainty_rate - fraction of lowest-uncertainty results to retain
    """
    results['index'] = range(len(results))
    # patients the model predicted correctly
    results = results[results['actual']==results['pred_binary']]
    # patients the model predicted with lower uncertainty
    results = results.sort_values(by='pred_raw_se')[0:int(len(results)*uncertainty_rate)]
    # randomly sample at a rate of 50% minority class, 50% majority class
    if len(results[results['pred_binary']==1])<num_positive_samples:
        print("Only {}<{} positive samples were identified".format(len(results[results['pred_binary']==1]), num_positive_samples))
        num_positive_samples = len(results[results['pred_binary']==1])

    results = results.sample(frac=1)
    positive_samples = results[results['pred_binary']==1][0:num_positive_samples]['index'].tolist()
    negative_samples = results[results['pred_binary']==0][0:num_positive_samples]['index'].tolist()
    samples = positive_samples + negative_samples
    samples = positive_samples
    print("Returning {} positive samples and {} negative samples".format(len(positive_samples), len(negative_samples)))

    return samples


def gnn_explainer(model, exp_loader, patient_list, params):
    print("Running GNNExplainer")

    outfile = '{}/{}_explainer'.format(params['outpath'], params['outname'])
    with open(f'{outfile}_edges.csv', 'w') as f_edge:
        f_edge.write("target_id,target_index,case,edge_index,edge_mask_value\n")
        with open(f'{outfile}_nodes.csv', 'w') as f_node:
            f_node.write("target_id,target_index,case,node_index,node_mask_value,{}\n".format(','.join('feature{}'.format(i) for i in range(params['num_features_longitudinal']))))

            counter = 0
            # assumes one graph is loaded at a time
            for data_batch in tqdm(exp_loader, total=len(patient_list)):
                # run for a node-level explanation (which nodes are important)
                # then run for a node-feature-level explanation (which features of those nodes are importance, which we aggregate across the time domain)
                # edge importance can be extracted from either of those (to determine prior/posterior comparison in edge weight importance)

                x_static_node, x_static_graph, x_longitudinal_node, x_longitudinal_graph, y, edge_index, edge_weight, batch, target_index = data_batch.patient_x_static.to(params['device']), data_batch.x.to(params['device']), data_batch.patient_x_longitudinal.to(params['device']), data_batch.x_longitudinal.to(params['device']), data_batch.y.unsqueeze(1).to(params['device']), data_batch.edge_index.to(params['device']), data_batch.edge_attr.to(params['device']), data_batch.batch.to(params['device']), data_batch.target_index.to(params['device'])
                x_longitudinal_graph_flat = torch.reshape(x_longitudinal_graph, (x_longitudinal_graph.shape[0],-1))

                kwargs = {'edge_weight':edge_weight, 'x_static_node':x_static_node, 'x_longitudinal_node':x_longitudinal_node, 'x_static_graph':x_static_graph, 'batch':batch, 'target_node':target_index, 'train':False}

                explainer = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=25),
                    explainer_config=dict(
                        explanation_type='model',
                        node_mask_type='object',
                        edge_mask_type='object'
                    ),
                    model_config=dict(
                        mode='classification',
                        task_level='graph',
                        return_type='log_probs',
                    ),
                )

                explanation = explainer(x=x_longitudinal_graph_flat, edge_index=edge_index, target=target_index, **kwargs)
                node_imp = explanation.node_mask.detach().cpu().tolist()

                explainer = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=25),
                    explainer_config=dict(
                        explanation_type='model',
                        node_mask_type='attributes',
                        edge_mask_type='object'
                    ),
                    model_config=dict(
                        mode='classification',
                        task_level='graph',
                        return_type='log_probs',
                    ),
                )

                explanation = explainer(x=x_longitudinal_graph_flat, edge_index=edge_index, target=target_index, **kwargs)
                edge_imp = explanation.edge_mask.detach().cpu().tolist()
                long_feat_imp = explanation.node_feat_mask.detach().cpu()
                # transform to format (num_nodes, num_years, num_features)
                feat_imp = torch.reshape(long_feat_imp,(long_feat_imp.shape[0],-1,params['num_features_longitudinal']))
                # aggregate along the time axis
                feat_imp_agr = feat_imp.mean(axis=1).tolist()

                target_id = patient_list[counter].item()
                target_index = target_index.detach().cpu().tolist()[0]
                case = y.detach().cpu().tolist()[0][0]
                node_index = range(len(node_imp))
                for node in node_index:
                    # write target id, target index, case, node index, node_mask_value, all the columns for node_feat_mask_value (one column for each feature)
                    node_data = [str(target_id), str(target_index), str(int(case)), str(node), str(round(node_imp[node],4))]
                    node_data.extend([str(round(i,4)) for i in feat_imp_agr[node]])
                    f_node.write(','.join(node_data)+'\n')

                edge_index = range(len(edge_imp))
                for edge in edge_index:
                    # write target id, target index, case, edge_index, edge_mask_value
                    edge_data = [str(target_id), str(target_index), str(int(case)), str(edge), str(round(edge_imp[edge],4))]
                    f_edge.write(','.join(edge_data)+'\n')

                counter += 1