import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F


class Baseline(torch.nn.Module):
    def __init__(self, num_features_static, hidden_dim, dropout_rate):
        super().__init__()
        self.static_linear1 = nn.Linear(num_features_static, hidden_dim)
        self.static_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_static):
        linear_out = self.relu(self.static_linear1(x_static))
        linear_out = self.relu(self.static_linear2(linear_out))
        linear_out = self.dropout(linear_out)
        out = self.sigmoid(self.final_linear(linear_out))
        return out


class BaselineLongitudinal(torch.nn.Module):
    def __init__(self, num_features_static, num_features_longitudinal, main_hidden_dim, lstm_hidden_dim, dropout_rate, num_lstm_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(num_features_longitudinal, lstm_hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True, bias=False)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.combined_linear1 = nn.Linear(num_features_static + lstm_hidden_dim*2, main_hidden_dim)
        self.combined_linear2 = nn.Linear(main_hidden_dim, main_hidden_dim)
        self.final_linear = nn.Linear(main_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_static, x_longitudinal):
        output, (hn, cn) = self.lstm(x_longitudinal)
        hn_bi = torch.cat((hn[0],hn[1]), dim=1)
        longitudinal_out = self.relu(hn_bi)
        x_cat = torch.cat((x_static, longitudinal_out), 1)
        linear_out = self.relu(self.combined_linear1(x_cat))
        linear_out = self.relu(self.combined_linear2(linear_out))
        linear_out = self.dropout(linear_out)
        out = self.sigmoid(self.final_linear(linear_out))
        return out


class GNN(torch.nn.Module):
    def __init__(self, num_features_static_graph, num_features_static_node, hidden_dim, gnn_layer, pooling_method, dropout_rate, ratio):
        super().__init__()
        self.pooling_method = pooling_method
        self.static_linear1 = nn.Linear(num_features_static_node, hidden_dim)
        self.static_linear2 = nn.Linear(hidden_dim, hidden_dim)

        # which gnn layer to use is specified by input argument
        if gnn_layer=='gcn':
            print("Using GCN layers")
            self.conv1 = gnn.GCNConv(num_features_static_graph, hidden_dim)
            self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        if gnn_layer=='graphconv':
            print("Using GraphConv layers")
            self.conv1 = gnn.GraphConv(num_features_static_graph, hidden_dim)
            self.conv2 = gnn.GraphConv(hidden_dim, hidden_dim)
        elif gnn_layer=='gat':
            print("Using GAT layers")
            self.conv1 = gnn.GATConv(num_features_static_graph, hidden_dim)
            self.conv2 = gnn.GATConv(hidden_dim, hidden_dim)

        self.pre_final_linear = nn.Linear(2*hidden_dim, hidden_dim)
        self.final_linear_com = nn.Linear(hidden_dim, 1)
        self.final_linear = nn.Linear(hidden_dim, 1)
        self.final_linear1 = nn.Linear(hidden_dim, 1)
        self.final_linear2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.TopKpool = gnn.TopKPooling(hidden_dim, ratio=ratio)
        self.SAGpool = gnn.SAGPooling(hidden_dim, ratio=ratio)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_static_node, x_static_graph, edge_index, edge_weight, batch, target_index):
        # patient part of the network
        linear_out = self.relu(self.static_linear1(x_static_node))
        linear_out = self.relu(self.static_linear2(linear_out))
        patient_out = self.dropout(linear_out)
        
        # family part of the network
        gnn_out = self.relu(self.conv1(x_static_graph, edge_index, edge_weight))
        gnn_out = self.relu(self.conv2(gnn_out, edge_index, edge_weight))

        if self.pooling_method=='target':
            out = gnn_out[target_index] # instead of pooling, use the target node embedding
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)
        elif self.pooling_method=='topkpool_sum':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='topkpool_mean':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_mean_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_sum':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_mean':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_mean_pool(out, pool_batch)

        family_out = self.dropout(out)
        
        # combined part of network (classifiation output)
        out = torch.cat((patient_out, family_out), 1)
        out = self.relu(self.pre_final_linear(out))
        out = self.sigmoid(self.final_linear_com(out))
        # separate output heads for different parts of the network (for loss calculations)
        patient_out = self.sigmoid(self.final_linear1(patient_out))
        family_out = self.sigmoid(self.final_linear2(family_out))
        return out, patient_out, family_out


class GNNLongitudinal(torch.nn.Module):
    def __init__(self, num_features_static_graph, num_features_static_node, num_features_longitudinal, main_hidden_dim, lstm_hidden_dim, gnn_layer, pooling_method, dropout_rate, ratio, num_lstm_layers=1):
        super().__init__()
        self.pooling_method = pooling_method
        self.lstm = nn.LSTM(num_features_longitudinal, lstm_hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True, bias=False)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.combined_linear1 = nn.Linear(num_features_static_node + lstm_hidden_dim*2, main_hidden_dim)
        self.combined_linear2 = nn.Linear(main_hidden_dim, main_hidden_dim)

        # which gnn layer to use is specified by input argument
        if gnn_layer=='gcn':
            print("Using GCN layers")
            self.combined_conv1 = gnn.GCNConv(num_features_static_graph + lstm_hidden_dim*2, main_hidden_dim)
            self.combined_conv2 = gnn.GCNConv(main_hidden_dim, main_hidden_dim)
        if gnn_layer=='graphconv':
            print("Using GraphConv layers")
            self.combined_conv1 = gnn.GraphConv(num_features_static_graph + lstm_hidden_dim*2, main_hidden_dim)
            self.combined_conv2 = gnn.GraphConv(main_hidden_dim, main_hidden_dim)
        elif gnn_layer=='gat':
            print("Using GAT layers")
            self.combined_conv1 = gnn.GATConv(num_features_static_graph + lstm_hidden_dim*2, main_hidden_dim)
            self.combined_conv2 = gnn.GATConv(main_hidden_dim, main_hidden_dim)

        self.pre_final_linear = nn.Linear(2*main_hidden_dim, main_hidden_dim)
        self.final_linear_com = nn.Linear(main_hidden_dim, 1)
        self.final_linear = nn.Linear(main_hidden_dim, 1)
        self.final_linear1 = nn.Linear(main_hidden_dim, 1)
        self.final_linear2 = nn.Linear(main_hidden_dim, 1)
        self.final_linear_lstm = nn.Linear(lstm_hidden_dim*2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.TopKpool = gnn.TopKPooling(main_hidden_dim, ratio=ratio)
        self.SAGpool = gnn.SAGPooling(main_hidden_dim, ratio=ratio)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_static_node, x_static_graph, x_longitudinal_node, x_longitudinal_graph, edge_index, edge_weight, batch, target_index):
        # patient part of the network
        output, (hn, cn) = self.lstm(x_longitudinal_node)
        hn_bi = torch.cat((hn[0],hn[1]), dim=1)
        longitudinal_out = self.relu(hn_bi)
        x_cat = torch.cat((x_static_node, longitudinal_out), 1)
        linear_out = self.relu(self.combined_linear1(x_cat))
        linear_out = self.relu(self.combined_linear2(linear_out))
        patient_out = self.dropout(linear_out)
        
        # family part of the network
        output, (hn, cn) = self.lstm(x_longitudinal_graph)
        hn_bi = torch.cat((hn[0],hn[1]), dim=1)
        longitudinal_out = self.relu(hn_bi)
        x_cat = torch.cat((x_static_graph, longitudinal_out), 1)
        gnn_out = self.relu(self.combined_conv1(x_cat, edge_index, edge_weight))
        gnn_out = self.relu(self.combined_conv2(gnn_out, edge_index, edge_weight))

        if self.pooling_method=='target':
            out = gnn_out[target_index] # instead of pooling, use the target node embedding
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)
        elif self.pooling_method=='topkpool_sum':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='topkpool_mean':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_mean_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_sum':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_mean':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_mean_pool(out, pool_batch)

        family_out = self.dropout(out)
        
        # combined part of network
        out = torch.cat((patient_out, family_out), 1)
        out = self.relu(self.pre_final_linear(out))
        out = self.sigmoid(self.final_linear_com(out))
        # separate parts of network
        patient_out = self.sigmoid(self.final_linear1(patient_out))
        family_out = self.sigmoid(self.final_linear2(family_out)) 
        lstm_out = self.sigmoid(self.final_linear_lstm(longitudinal_out[target_index])) 
        return out, patient_out, family_out, lstm_out


class GNNExplainabilityLSTM(torch.nn.Module):
    """This model directly explains the LSTM input, meaning that feature attributions are calculated for each feature at each time step
    This can be reduced to a single attribution for each feature by aggregating (e.g. averaging) across all time steps
    """
    def __init__(self, num_features_static_graph, num_features_static_node, num_features_longitudinal, main_hidden_dim, lstm_hidden_dim, gnn_layer, pooling_method, dropout_rate):
        super().__init__()
        num_lstm_layers = 1
        self.pooling_method = pooling_method
        self.lstm = nn.LSTM(num_features_longitudinal, lstm_hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True, bias=False)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.combined_linear1 = nn.Linear(num_features_static_node + lstm_hidden_dim*2, main_hidden_dim)
        self.combined_linear2 = nn.Linear(main_hidden_dim, main_hidden_dim)

        # which gnn layer to use is specified by input argument
        if gnn_layer=='gcn':
            print("Using GCN layers")
            self.combined_conv1 = gnn.GCNConv(num_features_static_graph + lstm_hidden_dim*2, main_hidden_dim)
            self.combined_conv2 = gnn.GCNConv(main_hidden_dim, main_hidden_dim)
        if gnn_layer=='graphconv':
            print("Using GraphConv layers")
            self.combined_conv1 = gnn.GraphConv(num_features_static_graph + lstm_hidden_dim*2, main_hidden_dim)
            self.combined_conv2 = gnn.GraphConv(main_hidden_dim, main_hidden_dim)
        elif gnn_layer=='gat':
            print("Using GAT layers")
            self.combined_conv1 = gnn.GATConv(num_features_static_graph + lstm_hidden_dim*2, main_hidden_dim)
            self.combined_conv2 = gnn.GATConv(main_hidden_dim, main_hidden_dim)

        self.pre_final_linear = nn.Linear(2*main_hidden_dim, main_hidden_dim)
        self.final_linear_com = nn.Linear(main_hidden_dim, 1)
        self.final_linear = nn.Linear(main_hidden_dim, 1)
        self.final_linear1 = nn.Linear(main_hidden_dim, 1)
        self.final_linear2 = nn.Linear(main_hidden_dim, 1)
        self.final_linear_lstm = nn.Linear(lstm_hidden_dim*2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.num_features_longitudinal = num_features_longitudinal
        
    def forward(self, x, edge_index, edge_weight=None, x_static_node=None, x_longitudinal_node=None, x_static_graph=None, batch=None, target_node=None, train=True):
        
        # NOTE x is a 2d tensor for x_longitudinal_graph but needs to be reshaped into 3d
        x_longitudinal_graph = torch.reshape(x, (x.shape[0],-1,self.num_features_longitudinal))

        # patient part of the network
        output, (hn, cn) = self.lstm(x_longitudinal_node)
        hn_bi = torch.cat((hn[0],hn[1]), dim=1)
        longitudinal_out = self.relu(hn_bi)
        x_cat = torch.cat((x_static_node, longitudinal_out), 1)
        linear_out = self.relu(self.combined_linear1(x_cat))
        linear_out = self.relu(self.combined_linear2(linear_out))
        patient_out = self.dropout(linear_out)
        
        # family part of the network
        output, (hn, cn) = self.lstm(x_longitudinal_graph)
        hn_bi = torch.cat((hn[0],hn[1]), dim=1)
        longitudinal_out = self.relu(hn_bi)
        x_cat = torch.cat((x_static_graph, longitudinal_out), 1)
        gnn_out = self.relu(self.combined_conv1(x_cat, edge_index, edge_weight))
        gnn_out = self.relu(self.combined_conv2(gnn_out, edge_index, edge_weight))

        if self.pooling_method=='target':
            out = gnn_out[target_node] # instead of pooling, use the target node embedding
        elif self.pooling_method=='sum':
            out = gnn.global_add_pool(gnn_out, batch)
        elif self.pooling_method=='mean':
            out = gnn.global_mean_pool(gnn_out, batch)
        elif self.pooling_method=='topkpool_sum':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='topkpool_mean':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.TopKpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_mean_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_sum':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_add_pool(out, pool_batch)
        elif self.pooling_method=='sagpool_mean':
            out, pool_edge_index, pool_edge_attr, pool_batch, _, _ = self.SAGpool(gnn_out, edge_index, edge_weight, batch)
            out = gnn.global_mean_pool(out, pool_batch)

        family_out = self.dropout(out)
        
        # combined part of network
        out = torch.cat((patient_out, family_out), 1)
        out = self.relu(self.pre_final_linear(out))
        out = self.sigmoid(self.final_linear_com(out))

        if train:
            # separate parts of network
            patient_out = self.sigmoid(self.final_linear1(patient_out))
            family_out = self.sigmoid(self.final_linear2(family_out)) 
            lstm_out = self.sigmoid(self.final_linear_lstm(longitudinal_out[target_node])) 
            return out, patient_out, family_out, lstm_out
        else:
            return out