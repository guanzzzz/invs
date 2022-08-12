import sys
import pickle
import numpy as np
import pandas as pd
import hyperopt as hp
from pprint import pprint
from Orange import preprocess
from Orange import projection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Orange.data import Table
from Orange.projection import manifold
from PySide6.QtCore import (QCoreApplication, QMetaObject)
from PySide6 import QtCore, QtGui, QtWidgets
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
# import hyperopt.pyll.stochastic


class InvPCA:
    def __init__(self):
        self.X = None
        self._X = None
        self.X_header = None
        self.Y = None
        self.preprocessor = None
        self.pca_model = None
        self.scores_df = None
        self.components_df = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.singular_values_ = None
        self.noise_variance_ = None
        self.covariance_df = None

    def set_data_and_model(self, csv_df, custom_parameter_dict=None):
        default_parameter_dict = {
            'n_components': None,
            'whiten': True,
            'svd_solver': 'auto',
            'random_state': None,
        }
        if custom_parameter_dict is None:
            pass
        else:
            default_parameter_dict.update(custom_parameter_dict)
        all_features = csv_df.columns.values.tolist()
        all_features.remove('class')
        self.X_header = all_features
        self.Y = csv_df['class']
        self.X = csv_df[self.X_header]
        self.pca_model = PCA(
            n_components=default_parameter_dict['n_components'],
            whiten=default_parameter_dict['whiten'],
            svd_solver=default_parameter_dict['svd_solver'],
            random_state=default_parameter_dict['random_state'],
        )

    def preprocess_data(self, selection=None):
        if selection is None:
            self.preprocessor = StandardScaler()
        self._X = self.preprocessor.fit_transform(self.X)
        # temp_df = pd.DataFrame(self._X)
        # temp_df.to_csv('scaled_origin_data.csv')

    def fit_transform(self):
        self.pca_model.fit(self._X)
        scores_array = self.pca_model.transform(self._X)
        self.scores_df = pd.DataFrame(scores_array)
        self.scores_df.to_csv('PCA_scores_df.csv')
        components = self.pca_model.components_
        self.components_df = pd.DataFrame(components)
        self.components_df.to_csv('PCA_components_df.csv')
        self.explained_variance = self.pca_model.explained_variance_
        self.explained_variance_ratio = self.pca_model.explained_variance_ratio_
        self.singular_values_ = self.pca_model.singular_values_
        self.noise_variance_ = self.pca_model.noise_variance_
        covariance_array = self.pca_model.get_covariance()
        self.covariance_df = pd.DataFrame(covariance_array)
        return self.scores_df, self.components_df, self.explained_variance, self.explained_variance_ratio


def stage_1_dataset_improvement(full_dataset, selection=None):
    """

    Predict target variable with required features and trained machine learning models

    :param full_dataset: full input dataset with all required features
    :param selection: prediction selection
    :return: prediciton: model prediction

    """
    if selection is None:
        pass
    else:
        selected_features = {
            'density': ['Co', 'pc', 'WC', 'gac', 'C', 'mpc', 'en', 'ven'],
            'k': ['Hco', 'dwc', 'mfp', 'Hwc', 'en', 'Cr3C2', 'ar', 'Vco',
                  'Con', 'pc', 'mpc', 'C'],
            'hardness': ['Hc', 'mfp', 'Hwc', 'Hco', 'dwc', 'ven', 'Vco', 'ar',
                         'pc', 'Con', 'Co', 'en'],
            'trs': ['Cr3C2', 'Hwc', 'Con', 'dwc', 'ar', 'Hco', 'ven', 'mfp',
                    'HV', 'en', 'Hc', 'pc'],
            'fracture': ['mfp', 'Hc', 'Hwc', 'Hco', 'ven', 'HV', 'ar', 'Con',
                         'pc', 'C', 'Vco', 'dwc']
        }

        '''
        trained_models = {
            'density': 'D_stacking_model.pkcls',
            'k': 'K_stacking_model.pkcls',
            'hardness': 'HV_stacking_model.pkcls',
            'fracture': 'KIC_stacking_model.pkcls',
            'trs': 'TRS_mlp_model.pkcls',
            'None': None
        }
        '''

        def prediction(full_input_dataset, target_selection):
            model_file = 'Model_Base.pkb'
            with open(model_file, 'rb') as qr:
                model = pickle.load(qr)
                selected_model = model[target_selection]
            input_dataset = full_input_dataset[selected_features[target_selection]]
            # print(input_dataset)
            direct_prediction = selected_model.predict(input_dataset.values)
            prediction_df = pd.DataFrame(direct_prediction, columns=[target_selection])
            return prediction_df

        return pd.DataFrame(prediction(full_dataset, selection), columns=[selection])


def stage_2_2_train_tsne_model(dataset_df, param_costom=None):
    """

    Train t-SNE model and get projection

    :param dataset_df: input dataset
    :param param_costom: custon t-SNE parameters
    :return: t-SNE projection

    """
    input_dataset = dataset_df
    param_default = {
        'settings_version': 4,
        'pca_components': 10,
        'normalize': True,
        'pca_projection': None,
        'affinities': None,
        'tsne_embedding': None,
        'iterations_done': None,
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': 200,
        'early_exaggeration_iter': 250,
        'early_exaggeration': 12,
        'n_iter': 750,
        'exaggeration': None,
        'theta': 0.5,
        'min_num_intervals': 10,
        'ints_in_interval': 1,
        'initialization': "pca",
        'metric': "euclidean",
        'n_jobs': 1,
        'neighbors': "exact",
        'negative_gradient_method': "bh",
        'multiscale': False,
        'callbacks': None,
        'callbacks_every_iters': 50,
        'random_state': None,
        'preprocessors': [preprocess.Normalize(),  preprocess.Continuize(),  preprocess.SklImpute()]
    }

    if param_costom is not None:
        param_default.update(param_costom)

    def pca_preprocessing(data, n_components, normalize):
        projector = projection.PCA(n_components=n_components, random_state=0)
        if normalize:
            projector.preprocessors += (preprocess.Normalize(),)

        model = projector(data)
        return model(data)

    tsne_model = manifold.TSNE(
        n_components=param_default['n_components'],
        perplexity=param_default['perplexity'],
        learning_rate=param_default['learning_rate'],
        early_exaggeration_iter=param_default['early_exaggeration_iter'],
        early_exaggeration=param_default['early_exaggeration'],
        n_iter=param_default['n_iter'],
        exaggeration=param_default['exaggeration'],
        theta=param_default['theta'],
        min_num_intervals=param_default['min_num_intervals'],
        ints_in_interval=param_default['ints_in_interval'],
        initialization=param_default['initialization'],
        metric=param_default['metric'],
        n_jobs=param_default['n_jobs'],
        neighbors=param_default['neighbors'],
        negative_gradient_method=param_default['negative_gradient_method'],
        multiscale=param_default['multiscale'],
        callbacks=param_default['callbacks'],
        callbacks_every_iters=param_default['callbacks_every_iters'],
        random_state=param_default['random_state'],
        preprocessors=param_default['preprocessors'],
    )

    input_dataset = Table(input_dataset)
    input_dataset = pca_preprocessing(input_dataset, param_default['pca_components'], param_default['normalize'])
    tsne_proj = tsne_model.fit(input_dataset.X)
    header = ['tsne-dim-{}'.format(str(i+1)) for i in range(param_default['n_components'])]
    tsne_proj = pd.DataFrame(tsne_proj, columns=header)
    # tsne_proj.to_csv('tsne_projection.csv')
    return tsne_proj


def stage_2_2_calculate_distance_and_rank(x_df, y_df, key_data_number=10):
    """

    Find most similar data using t-SNE 2-dimensional data

    :param x_df: t-SNE x-axis dataframe
    :param y_df: t-SNE y-axis dataframe
    :param key_data_number: number of data points considered for calculation; default=10
    :return: row number of data ranked from high to low according to the calculated similarity

    """
    distance_rank_dict = {}
    i = 0

    for row_y in y_df.itertuples():
        y_x = row_y[1]
        y_y = row_y[2]
        z = []

        for row_x in x_df.itertuples():
            x_x = row_x[1]
            x_y = row_x[2]
            z.append(np.power((y_x - x_x), 2) + np.power((y_y - x_y), 2))

        column_id = 'distance {}'.format(str(i))
        sort_dict = {
            column_id: z,
            'No': x_df['No'].values.ravel().tolist()
        }
        z_df = pd.DataFrame.from_dict(sort_dict)
        z_df = z_df.sort_values(by=column_id, ascending=True)
        distance_df = z_df['No']
        distance_rank_dict['{}'.format(str(i + 1))] = distance_df.head(key_data_number).values.ravel()
        i += 1

    return pd.DataFrame.from_dict(distance_rank_dict)


def stage_2_3_pca_modeling(dataset_df, custom_param_dict=None):
    """

    Train and save a pca model

    :param custom_param_dict: pca parameter dict
    :param dataset_df: input dataset dataframe
    :return pca_model: trained pca model

    """

    pca_model = InvPCA()
    pca_model.set_data_and_model(dataset_df, custom_param_dict)
    pca_model.preprocess_data()
    with open('pca_model.pkp', 'wb') as qr:
        pickle.dump(pca_model, qr)
    return pca_model


def stage_3_1_calculate_ellipse(scores_df, z_df):
    """

    Calculate the ellipse for Bayesian searching

    :param scores_df: PCA scores dataframe
    :param z_df: distance dataframe calculated by 'stage_2_1_calculate_distance_and_rank' method
    :return: ellipse_list: a list of calculated ellipse

    """
    # print('z_df', z_df)
    # print('scores_df', scores_df)
    header_list = z_df.columns.values.tolist()
    header_list = header_list[1:]
    ellipse_list = []

    for i in range(len(header_list)):
        sub_z_list = z_df[header_list[i]].values.ravel()
        ellipse_df = pd.DataFrame()

        for j in range(len(sub_z_list)):
            temp_df = scores_df[scores_df.No == sub_z_list[j]]
            ellipse_df = pd.concat([ellipse_df, temp_df], axis=0)

        ellipse_df_header = ellipse_df.columns.values.tolist()
        ellipse_df_header.remove('No')
        # print('ellipse_df_header', ellipse_df_header)
        # print('ellipse_df', ellipse_df)
        ellipse_statical_summary = []

        for j in range(len(ellipse_df_header)):
            temp_array = ellipse_df[ellipse_df_header[j]].values.ravel()
            ellipse_statical = [ellipse_df_header[j], temp_array.mean(), temp_array.var()]
            ellipse_statical_summary.append(ellipse_statical)

        # ellipse_df.to_csv('ellipse-No-{}.csv'.format(str(i + 1)))
        ellipse_statical_df = pd.DataFrame(ellipse_statical_summary)
        # ellipse_statical_df.to_csv('ellipse-statical-No-{}.csv'.format(str(i + 1)))
        ellipse_list.append(ellipse_statical_df)

    return ellipse_list


def stage_3_2_extract_origin_property(property_df):
    """

    Set origin properties as target variable and extract

    :param property_df: origin properties dataframe
    :return:

    """
    i = 1
    header = property_df.columns.values[:]
    # print(header)
    property_df_list = []
    for row in property_df.itertuples():
        temp_property_array = [list(row[1:])]
        # print(temp_property_array)
        temp_property_df = pd.DataFrame(temp_property_array, columns=header)
        property_df_list.append(temp_property_df)
        # temp_property_df.to_csv('origin_property_No_{}.csv'.format(i))
        i += 1

    return property_df_list


class Stage4Thread(QtCore.QObject):
    Output_signal = QtCore.Signal(object, object, int, int, int)

    def __init__(self):
        super().__init__()

    def stage_4_inv_pca_search(self, number, pca_model, preprocessor, ellipse_statical_df,
                               origin_property_df, selected_idx_list,
                               search_times, selected_pcs):
        """

        Bayesian searching with desired properties

        :param number: number of searching tasks
        :param pca_model: trained PCA model
        :param preprocessor: corresponding dataset preprocessor
        :param ellipse_statical_df: calculated ellipse
        :param origin_property_df: original property dataframe
        :param selected_idx_list: column index of selected property
        :param search_times: searching times
        :param selected_pcs: selected principal components
        :return inverse_df: inverse prediction data
        :return tpe_data: searching record

        """
        origin_property_header = origin_property_df.columns.values
        origin_property_array = origin_property_df[origin_property_header[:]].values

        '''
        def dict_to_array(**params):
            array = []
            for key in params.keys():
                array.append(params[key])
            return array
        '''

        def calculate_difference(params):
            """

            Definition of loss function

            """
            print(params)
            pcs = []
            for j in range(selected_pcs):
                pcs.append(params['{}'.format(j)])
            temp_inv_array = pca_model.inverse_transform(pcs)
            temp_inv_prep_array = preprocessor.inverse_transform(temp_inv_array.reshape(1, -1))
            distance = 0
            for j in range(len(selected_idx_list)):
                distance += np.true_divide(
                    np.power((temp_inv_prep_array[0][selected_idx_list[j]] - origin_property_array[0][j]), 2),
                    np.power(origin_property_array[0][j], 2)
                )
            self.Output_signal.emit(distance, '', 0, 0, number)
            return {'loss': distance, 'status': STATUS_OK}
        space = {}
        i = 0
        # print("ellipse_statical_df:", ellipse_statical_df)
        for row in ellipse_statical_df.itertuples():
            space[str(i)] = hp.uniform('{}'.format(i),
                                       row[2] - 1.4 * np.sqrt(row[3]),
                                       row[2] + 1.4 * np.sqrt(row[3]))
            i += 1

        trials = Trials()
        best = fmin(calculate_difference,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=search_times,
                    trials=trials)
        # print(best)
        tpe_data = pd.DataFrame({
            'loss': [x['loss'] for x in trials.results],
        })
        # save searching record
        for i in range(selected_pcs):
            temp_pc = pd.DataFrame({'PC{}'.format(str(i + 1)): trials.idxs_vals[1]['{}'.format(str(i))]})
            tpe_data = pd.concat([tpe_data, temp_pc], axis=1)

        # tpe_data.to_csv('tpe_search.csv')
        inverse_df = pd.DataFrame()
        # inverse prediction and preprocess
        for row in tpe_data.itertuples():
            inv_pca_array = row[2:]
            inv_array = pca_model.inverse_transform(inv_pca_array)
            inv_prep_array = preprocessor.inverse_transform(inv_array.reshape(1, -1))
            temp_df = pd.DataFrame(inv_prep_array)
            inverse_df = pd.concat([inverse_df, temp_df], axis=0)

        # print(inverse_df)
        inverse_df = pd.concat([inverse_df.reset_index(), pd.DataFrame(tpe_data['loss'])], axis=1)
        # inverse_df.to_csv('tpe_search_inverse_pca.csv')

        self.Output_signal.emit(inverse_df, tpe_data, 1, 1, number)


def table_view(table_widget, input_table):
    input_table_rows = input_table.shape[0]
    input_table_columns = input_table.shape[1]
    model = QtGui.QStandardItemModel(input_table_rows, input_table_columns)
    model.setHorizontalHeaderLabels([str(i) for i in input_table.columns.values.tolist()])
    for i in range(input_table_rows):
        input_table_rows_values = input_table.iloc[[i]]
        input_table_rows_values_array = np.array(input_table_rows_values)
        input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
        for j in range(input_table_columns):
            input_table_items_list = input_table_rows_values_list[j]
            input_table_items = str(input_table_items_list)
            newItem = QtGui.QStandardItem(input_table_items)
            model.setItem(i, j, newItem)
    table_widget.setModel(model)
    return model


class UiForm(object):
    def setupUi(self, form):
        if not form.objectName():
            form.setObjectName(u"form")
        form.resize(400, 300)
        self.retranslateUi(form)
        QMetaObject.connectSlotsByName(form)

    # setupUi and retranslate Ui
    def retranslateUi(self, form):
        form.setWindowTitle(QCoreApplication.translate("form", u"form", None))


class Main(QtWidgets.QWidget, UiForm):
    """

    A simple UI for applying the strategy

    :param self.stg1_init_csv: dataset to be improved with all corresponding features. required file format: (.csv)
    :param self.stg2_csv: training and testing dataset with label column named 'class'. required file format: (.csv)
    :param self.stg2_x_csv: training data with t-SNE decomposed dimension
    :param self.stg1_y_csv: target data with t-SNE decomposed dimension
    :param self.stg1_distance_Z_df:
    :param self.stg2_pca_model: trained PCA model
    :param self.stg2_pca_model: loaded trained PCA model. required file format: (.pkl)
    :param self.stg2_distance_Z_df: calculated distance dataframe
    :param self.stg3_origin_property_df: loaded original properties. required file format: (.csv)
    :param self.stg3_ellipse_statical_df:

    """
    stg4_param = QtCore.Signal(int, object, object, object, object, object, int, int)

    def __init__(self):
        super(Main, self).__init__()
        self.stg1_init_csv = None
        self.stg2_csv = None
        self.stg2_x_csv = None
        self.stg2_y_csv = None
        self.stg2_distance_Z_df = None
        self.stg2_pca_model = None
        self.stg3_ellipse_statical_df_list = None
        self.stg3_origin_property_df = None
        self.stg3_extracted_origin_property = None
        self.stg4_task_thread = QtCore.QThread(self)
        self.stg4_processor = Stage4Thread()
        self.stg4_processor.moveToThread(self.stg4_task_thread)
        self.stg4_param.connect(self.stg4_processor.stage_4_inv_pca_search)
        self.stg4_processor.Output_signal.connect(self.update_searching_status)
        self.table_widget_model = {}
        self.table_widgets = {}
        self.global_layout = QtWidgets.QVBoxLayout(self)

        # Stage 1 dataset improvement
        # Open csv file
        self.stg1_open_init_csv_btn = QtWidgets.QPushButton()
        self.stg1_open_init_csv_btn.setText("Stage 1-1 Open Init Dataset")
        self.global_layout.addWidget(self.stg1_open_init_csv_btn)
        self.stg1_open_init_csv_btn.clicked.connect(self.stage_1_1_load_init_csv)
        # Select target property
        self.stg1_improved_rdo_btn = [
            QtWidgets.QRadioButton(),
            QtWidgets.QRadioButton(),
            QtWidgets.QRadioButton(),
            QtWidgets.QRadioButton(),
            QtWidgets.QRadioButton(),
        ]

        self.stg1_predict_choice_groupbox = QtWidgets.QGroupBox()
        self.stg1_predict_choice_groupbox.setTitle('Select Target Property')
        self.stg1_sub_layout = QtWidgets.QHBoxLayout()
        self.global_layout.addWidget(self.stg1_predict_choice_groupbox)
        self.stg1_predict_choice_groupbox.setLayout(self.stg1_sub_layout)
        stg1_rdo_name = ['D', 'K', 'HV', 'TRS', 'KIC']
        self.stg1_improved_rdo_btn[0].setChecked(True)
        for i in range(len(stg1_rdo_name)):
            self.stg1_improved_rdo_btn[i].setText(stg1_rdo_name[i])
            self.stg1_sub_layout.addWidget(self.stg1_improved_rdo_btn[i])
        # Predict target property
        self.stg1_predict_btn = QtWidgets.QPushButton()
        self.stg1_predict_btn.setText("Stage 1-2 Prediction")
        self.global_layout.addWidget(self.stg1_predict_btn)
        self.stg1_predict_btn.clicked.connect(self.stage_1_2_prediction)

        # Stage 2 train t-SNE model and PCA model
        # Open csv file
        self.stg2_open_csv_btn = QtWidgets.QPushButton()
        self.stg2_open_csv_btn.setText("Stage 2-1 Open Csv")
        self.global_layout.addWidget(self.stg2_open_csv_btn)
        self.stg2_open_csv_btn.clicked.connect(self.stage_2_1_open_csv)

        # Train t-SNE model
        self.stg2_compute_tsne_btn = QtWidgets.QPushButton()
        self.stg2_compute_tsne_btn.setText("Stage 2-2 t-SNE Modeling and Ranking")
        self.global_layout.addWidget(self.stg2_compute_tsne_btn)
        self.stg2_compute_tsne_btn.clicked.connect(self.stage_2_2_tsne)

        # Train PCA model
        self.stg2_pca_parameter_groupbox = QtWidgets.QGroupBox()
        self.stg2_pca_parameter_groupbox.setTitle('PCA Parameter Setting')
        self.global_layout.addWidget(self.stg2_pca_parameter_groupbox)
        self.stg2_pca_parameter_groupbox_layout = QtWidgets.QVBoxLayout()
        self.stg2_pca_parameter_groupbox.setLayout(self.stg2_pca_parameter_groupbox_layout)

        self.stg2_pca_n_components_label_layout = QtWidgets.QHBoxLayout()
        self.stg2_pca_n_components_label = QtWidgets.QLabel()
        self.stg2_pca_n_components_label.setText('PCA Components:')
        self.stg2_pca_n_components_line_edit = QtWidgets.QLineEdit()
        self.stg2_pca_n_components_line_edit.setText('None')
        self.stg2_pca_n_components_line_edit.setValidator(QtGui.QIntValidator())
        self.stg2_pca_n_components_label_layout.addWidget(self.stg2_pca_n_components_label)
        self.stg2_pca_n_components_label_layout.addWidget(self.stg2_pca_n_components_line_edit)
        self.stg2_pca_parameter_groupbox_layout.addLayout(self.stg2_pca_n_components_label_layout)

        self.stg2_pca_whiten_label_layout = QtWidgets.QHBoxLayout()
        self.stg2_svd_solver_label = QtWidgets.QLabel()
        self.stg2_svd_solver_label.setText('SVD Solver:')
        self.stg2_svd_combo_box = QtWidgets.QComboBox()
        self.stg2_svd_combo_box.addItems(['Auto', 'Full', 'Arpack', 'Randomized'])
        self.stg2_pca_whiten_check_box = QtWidgets.QCheckBox()
        self.stg2_pca_whiten_check_box.setText('Whiten')
        self.stg2_pca_whiten_check_box.setChecked(True)
        self.stg2_pca_whiten_label_layout.addWidget(self.stg2_svd_solver_label)
        self.stg2_pca_whiten_label_layout.addWidget(self.stg2_svd_combo_box)
        self.stg2_pca_whiten_label_layout.addWidget(self.stg2_pca_whiten_check_box)
        self.stg2_pca_parameter_groupbox_layout.addLayout(self.stg2_pca_whiten_label_layout)

        self.stg2_pca_random_state_layout = QtWidgets.QHBoxLayout()
        self.stg2_pca_random_state_label = QtWidgets.QLabel()
        self.stg2_pca_random_state_label.setText('Random State:')
        self.stg2_pca_random_state_line_edit = QtWidgets.QLineEdit()
        self.stg2_pca_random_state_line_edit.setText('None')
        self.stg2_pca_random_state_line_edit.setValidator(QtGui.QDoubleValidator())
        self.stg2_pca_random_state_layout.addWidget(self.stg2_pca_random_state_label)
        self.stg2_pca_random_state_layout.addWidget(self.stg2_pca_random_state_line_edit)
        self.stg2_pca_parameter_groupbox_layout.addLayout(self.stg2_pca_random_state_layout)

        self.stg2_compute_pca_btn = QtWidgets.QPushButton()
        self.stg2_compute_pca_btn.setText("Stage 2-3 PCA Modeling")
        self.global_layout.addWidget(self.stg2_compute_pca_btn)
        self.stg2_compute_pca_btn.clicked.connect(self.stage_2_3_pca)

        # Stage 3 determine searching space
        # calculate ellipse with trained pca model
        self.stg3_calculate_ellipse_btn = QtWidgets.QPushButton()
        self.stg3_calculate_ellipse_btn.setText("Stage 3-1 Calculate Searching Space")
        self.global_layout.addWidget(self.stg3_calculate_ellipse_btn)
        self.stg3_calculate_ellipse_btn.clicked.connect(self.stage_3_1_calculate_ellipse)

        # Load origin property csv and extract
        self.stg3_transform_origin_property_all_btn = QtWidgets.QPushButton()
        self.stg3_transform_origin_property_all_btn.setText("Stage 3-2 Transform Origin Property")
        self.global_layout.addWidget(self.stg3_transform_origin_property_all_btn)
        self.stg3_transform_origin_property_all_btn.clicked.connect(self.stage_3_2_transform_origin_property)

        # Stage 4 Bayesian searching
        # Searching for potential candidates within pre-defined searching space and reverse prediction
        self.stg4_searching_times_groupbox = QtWidgets.QGroupBox()
        self.stg4_searching_times_groupbox.setTitle('Bayesian Searching Setting')
        self.stg4_searching_times_layout = QtWidgets.QHBoxLayout()
        self.stg4_searching_times_label = QtWidgets.QLabel()
        self.stg4_searching_times_label.setText('Searching Times:')
        self.stg4_searching_times_combo_box = QtWidgets.QComboBox()
        self.stg4_searching_times_combo_box.addItems(['20', '100', '200', '500', '1000', '2500', '5000', '10000'])
        self.stg4_searching_times_combo_box.setCurrentIndex(6)
        self.stg4_searching_times_layout.addWidget(self.stg4_searching_times_label)
        self.stg4_searching_times_layout.addWidget(self.stg4_searching_times_combo_box)
        self.stg4_searching_times_groupbox.setLayout(self.stg4_searching_times_layout)
        self.global_layout.addWidget(self.stg4_searching_times_groupbox)

        self.stg4_begin_search_btn = QtWidgets.QPushButton()
        self.stg4_begin_search_btn.setText("Stage 4 Bayesian Searching and Reverse Prediction")
        self.global_layout.addWidget(self.stg4_begin_search_btn)
        self.stg4_begin_search_btn.clicked.connect(self.stage_4_reverse_prediction)

        self.infobox = QtWidgets.QLabel()
        self.global_layout.addWidget(self.infobox)

    @staticmethod
    def event_open_file(file_type, layout, table_widget, infobox):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            file_diag = QtWidgets.QFileDialog()
            position = QtWidgets.QFileDialog.getOpenFileName(
                file_diag, 'select file', '', file_type, options=options)
            if isinstance(position, tuple):
                loaded_file = position[0]
                if loaded_file == '':
                    return None, None
                else:
                    try:
                        layout.removeWidget(table_widget)
                    except Exception as e:
                        print(e)
                    table_widget.setSizePolicy(sizePolicy)
                    table_widget.setSizeIncrement(QtCore.QSize(0, 20))
                    layout.addWidget(table_widget)
                    container = pd.read_csv(loaded_file)
                    infobox.setText(loaded_file)
                    model = table_view(table_widget, container)
                    return model, container
            else:
                return None, None
        except Exception as e:
            infobox.setText(str(e))
            return None, None

    @QtCore.Slot()
    def stage_1_1_load_init_csv(self):
        self.table_widgets['stage_1_load_init_csv_widget'] = QtWidgets.QTableView()
        mark = QtWidgets.QLabel()
        mark.setText("Stage 1 Init Csv")
        self.global_layout.addWidget(mark)
        self.table_widget_model['stage_1_load_init_csv_model'], self.stg1_init_csv =\
            self.event_open_file('csv files(*.csv)', self.global_layout,
                                 self.table_widgets['stage_1_load_init_csv_widget'],
                                 self.infobox)

    @QtCore.Slot()
    def stage_1_2_prediction(self):
        if self.stg1_init_csv is None:
            self.infobox.setText('Please load a dataset')
        else:
            try:
                if self.stg1_improved_rdo_btn[0].isChecked():
                    prediction_df = stage_1_dataset_improvement(self.stg1_init_csv, 'density')
                    prediction_df.to_csv('density_prediction.csv')
                elif self.stg1_improved_rdo_btn[1].isChecked():
                    prediction_df = stage_1_dataset_improvement(self.stg1_init_csv, 'k')
                    prediction_df.to_csv('coercive_force_prediction.csv')
                elif self.stg1_improved_rdo_btn[2].isChecked():
                    prediction_df = stage_1_dataset_improvement(self.stg1_init_csv, 'hardness')
                    prediction_df.to_csv('hardness_prediction.csv')
                elif self.stg1_improved_rdo_btn[3].isChecked():
                    prediction_df = stage_1_dataset_improvement(self.stg1_init_csv, 'trs')
                    prediction_df.to_csv('transverse_rupture_strength_prediction.csv')
                else:
                    prediction_df = stage_1_dataset_improvement(self.stg1_init_csv, 'fracture')
                    prediction_df.to_csv('fracture_toughness_prediction.csv')
                self.infobox.setText('Prediction complete')
            except Exception as e:
                self.infobox.setText(str(e))

    @QtCore.Slot()
    def stage_2_1_open_csv(self):
        self.table_widgets['stage_2_load_init_csv_widget'] = QtWidgets.QTableView()
        mark = QtWidgets.QLabel()
        mark.setText("Stage 2 Init Csv")
        self.global_layout.addWidget(mark)
        self.table_widget_model['stage_2_load_init_csv_model'], self.stg2_csv = \
            self.event_open_file('csv files(*.csv)', self.global_layout,
                                 self.table_widgets['stage_2_load_init_csv_widget'],
                                 self.infobox)

    @QtCore.Slot()
    def stage_2_2_tsne(self):
        if self.stg2_csv is None:
            self.infobox.setText('Please load a dataset')
        else:
            try:
                stg2_init_csv_label = self.stg2_csv['class']
                stg2_init_csv_without_label = self.stg2_csv.drop(columns=['class'], inplace=False)
                stg2_tsne_proj_without_label = stage_2_2_train_tsne_model(stg2_init_csv_without_label,
                                                                          param_costom=None)
                stg2_tsne_proj_with_label = pd.concat([stg2_tsne_proj_without_label, stg2_init_csv_label], axis=1)
                stg2_x_csv_without_number = stg2_tsne_proj_with_label[
                    stg2_tsne_proj_with_label['class'].str.contains('train')]
                stg2_x_csv_without_number = stg2_x_csv_without_number.reset_index(drop=True)
                stg2_y_csv = stg2_tsne_proj_with_label[stg2_tsne_proj_with_label['class'].str.contains('test')]
                stg2_x_row_number = stg2_x_csv_without_number.shape[0]
                stg2_x_number_list = [str(i+1) for i in range(stg2_x_row_number)]
                stg2_x_number_df = pd.DataFrame(stg2_x_number_list, columns=['No'])
                self.stg2_x_csv = pd.concat([stg2_x_csv_without_number, stg2_x_number_df], axis=1)
                self.stg2_x_csv = self.stg2_x_csv.drop(columns=['class'], axis=1, inplace=False)
                self.stg2_y_csv = stg2_y_csv.drop(columns=['class'], axis=1, inplace=False)
                stg2_tsne_proj_with_label.to_csv('t-SNE_projection.csv')
                self.stg2_distance_Z_df = stage_2_2_calculate_distance_and_rank(self.stg2_x_csv, self.stg2_y_csv,
                                                                                key_data_number=10)
                with open('t-SNE_distance.pkd', 'wb') as qr:
                    pickle.dump(self.stg2_distance_Z_df, qr)
            except Exception as e:
                self.infobox.setText(str(e))

    @QtCore.Slot()
    def stage_2_3_pca(self):
        if self.stg2_csv is None:
            self.infobox.setText('Please load a dataset')
        else:
            try:
                default_parameter_dict = {
                    'n_components': None,
                    'whiten': self.stg2_pca_whiten_check_box.isChecked(),
                    'svd_solver': self.stg2_svd_combo_box.currentText(),
                    'random_state': None,
                }
                try:
                    custom_parameter_dict = {
                        'n_components': eval(self.stg2_pca_n_components_line_edit.text()),
                        'random_state': eval(self.stg2_pca_random_state_line_edit.text())
                    }
                    default_parameter_dict.update(custom_parameter_dict)
                except Exception as e:
                    self.infobox.setText(str(e))

                if default_parameter_dict['n_components'] is not None:
                    if default_parameter_dict['n_components'] <= 0 or \
                            default_parameter_dict['n_components'] > self.stg2_csv.shape[1]-1:
                        default_parameter_dict['n_components'] = None
                        self.stg2_pca_n_components_line_edit.setText('None')
                    else:
                        pass

                print(default_parameter_dict)
                stg2_pca_init_csv_label = self.stg2_csv['class']
                # stg2_pca_init_csv_without_label = self.stg2_csv.drop(columns=['class'], inplace=False)
                self.stg2_pca_model = stage_2_3_pca_modeling(self.stg2_csv, default_parameter_dict)
                stg2_scores_df_without_header, stg2_components_df_without_header,\
                    stg2_explained_variance_array, stg2_explained_variance_ratio_array =\
                    self.stg2_pca_model.fit_transform()
                with open('PCA_model.pkp', 'wb') as qr:
                    pickle.dump(self.stg2_pca_model, qr)
                stg2_scores_df_header = [
                    'PC{}'.format(str(i+1)) for i in range(len(stg2_scores_df_without_header.columns.values.tolist()))]
                stg2_scores_df_without_label = pd.DataFrame(stg2_scores_df_without_header.values,
                                                            columns=stg2_scores_df_header)
                stg2_scores_df_with_label = pd.concat([stg2_scores_df_without_label, stg2_pca_init_csv_label], axis=1)
                stg2_scores_df_with_label.to_csv('PCA_scores_df.csv')

                stg2_components_df_part = pd.DataFrame(stg2_components_df_without_header.values,
                                                       columns=stg2_scores_df_header)
                stg2_pcs_df = pd.DataFrame(stg2_scores_df_header, columns=['Principal Components'])
                stg2_components_df_full = pd.concat([stg2_components_df_part, stg2_pcs_df], axis=1)
                stg2_components_df_full.to_csv('PCA_components_df.csv')

                stg2_explained_variance_df = pd.DataFrame(stg2_explained_variance_array, columns=['Explained Variance'])
                stg2_explained_variance_ratio_df = pd.DataFrame(
                    stg2_explained_variance_ratio_array, columns=['Explained Variance Ratio'])
                stg2_scree_df = pd.concat([stg2_explained_variance_df, stg2_explained_variance_ratio_df, stg2_pcs_df],
                                          axis=1)
                stg2_scree_df.to_csv('PCA_scree.csv')
            except Exception as e:
                self.infobox.setText(str(e))

    @QtCore.Slot()
    def stage_3_1_calculate_ellipse(self):
        pca_status = 0
        ellipse_z_status = 0
        if self.stg2_pca_model is None:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                file_diag = QtWidgets.QFileDialog()
                pca_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'Select File', '',
                                                                     'Pickled PCA Model(*.pkp)', options=options)
                if isinstance(pca_position, tuple):
                    pca_file = pca_position[0]
                    if pca_file == '':
                        pass
                    else:
                        with open(pca_file, 'rb') as qr:
                            self.stg2_pca_model = pickle.load(qr)
                        self.infobox.setText(pca_file)
                        pca_status = 1
                else:
                    pass
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pca_status = 1

        if self.stg2_distance_Z_df is None:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                file_diag = QtWidgets.QFileDialog()
                distance_z_file_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'Select File', '',
                                                                                 'Pickled Distance Data(*.pkd)',
                                                                                 options=options)
                if isinstance(distance_z_file_position, tuple):
                    distance_z_file = distance_z_file_position[0]
                    if distance_z_file == '':
                        pass
                    else:
                        with open(distance_z_file, 'rb') as qr:
                            self.stg2_distance_Z_df = pickle.load(qr)
                        self.infobox.setText(distance_z_file)
                        ellipse_z_status = 1
                else:
                    pass
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            ellipse_z_status = 1

        if pca_status == 1 and ellipse_z_status == 1:

            scores_df_without_number = self.stg2_pca_model.scores_df
            scores_df_number_list = [str(i+1) for i in range(scores_df_without_number.shape[0])]
            scores_df_number_df = pd.DataFrame(scores_df_number_list, columns=['No'])
            scores_df_with_number = pd.concat([scores_df_without_number, scores_df_number_df], axis=1)
            # print(scores_df_with_number)
            self.stg3_ellipse_statical_df_list = stage_3_1_calculate_ellipse(
                scores_df_with_number, self.stg2_distance_Z_df)
            pprint(self.stg3_ellipse_statical_df_list)
            self.stg3_ellipse_statical_df_list[0].to_csv('test.csv')
            with open('Ellipse_statical.pks', 'wb') as qr:
                pickle.dump(self.stg3_ellipse_statical_df_list, qr)

    @QtCore.Slot()
    def stage_3_2_transform_origin_property(self):
        if self.stg2_csv is None:
            self.infobox.setText("Please load the appropriate dataset using 'Stage 2-1 Open Csv' Button")
        else:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                file_diag = QtWidgets.QFileDialog()
                csv_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'Select File', '',
                                                                     'Csv File (*.csv)', options=options)
                if isinstance(csv_position, tuple):
                    csv_file = csv_position[0]
                    print("csv_file:{}".format(csv_file))
                    if csv_file == '':
                        pass
                    else:
                        self.stg3_origin_property_df = pd.read_csv(csv_file)
                        self.infobox.setText(csv_file)
                        origin_dataset_header = self.stg2_csv.columns.tolist()
                        origin_property_header = self.stg3_origin_property_df.columns.tolist()
                        origin_property_summary = {}
                        origin_property_header_list = []
                        for i in range(len(origin_dataset_header)):

                            for j in range(len(origin_property_header)):

                                if origin_dataset_header[i] == origin_property_header[j]:
                                    origin_property_header_list.append(i)
                                else:
                                    pass
                        origin_property_summary['origin_property_header_list'] = origin_property_header_list

                        origin_property_df_list = []
                        for i in range(self.stg3_origin_property_df.shape[0]):
                            temp_df = pd.DataFrame(self.stg3_origin_property_df.iloc[i:i+1, :],
                                                   columns=origin_property_header)
                            origin_property_df_list.append(temp_df)
                        origin_property_summary['origin_property_df_list'] = origin_property_df_list

                        self.stg3_extracted_origin_property = origin_property_summary
                        with open('original_property.pko', 'wb') as qr:
                            pickle.dump(origin_property_summary, qr)
                else:
                    pass
            except Exception as e:
                self.infobox.setText(str(e))

    def update_searching_status(self, obj1, obj2, flag1, flag2, number):
        if flag1 == 0:
            self.infobox.setText('No {s1} Searching Task :: Current Loss :: {s2}'.format(s1=str(number), s2=str(obj1)))
            self.stg4_begin_search_btn.setDisabled(True)
        else:
            if flag2 == 1:
                self.stg4_begin_search_btn.setDisabled(False)
            obj1.to_csv('Searching_results_No{}.csv'.format(str(number + 1)))
            obj2.to_csv('Searching_records_No{}.csv'.format(str(number + 1)))

    @QtCore.Slot()
    def stage_4_reverse_prediction(self):
        pca_status = 0
        ellipse_statical_status = 0
        origin_property_status = 0

        if self.stg2_pca_model is None:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                file_diag = QtWidgets.QFileDialog()
                pca_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'Select File', '',
                                                                     'Pickled PCA Model(*.pkp)', options=options)
                if isinstance(pca_position, tuple):
                    pca_file = pca_position[0]
                    if pca_file == '':
                        pass
                    else:
                        with open(pca_file, 'rb') as qr:
                            self.stg2_pca_model = pickle.load(qr)
                        self.infobox.setText(pca_file)
                        pca_status = 1
                else:
                    pass
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pca_status = 1

        if self.stg3_ellipse_statical_df_list is None:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                file_diag = QtWidgets.QFileDialog()
                ellipse_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'Select File', '',
                                                                         'Pickled Ellipse statical data(*.pks)',
                                                                         options=options)
                if isinstance(ellipse_position, tuple):
                    ellipse_file = ellipse_position[0]
                    if ellipse_file == '':
                        pass
                    else:
                        with open(ellipse_file, 'rb') as qr:
                            self.stg3_ellipse_statical_df_list = pickle.load(qr)
                        self.infobox.setText(ellipse_file)
                        ellipse_statical_status = 1
                else:
                    pass
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            ellipse_statical_status = 1

        if self.stg3_extracted_origin_property is None:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                file_diag = QtWidgets.QFileDialog()
                property_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'Select File', '',
                                                                          'Pickled Origin Property(*.pko)',
                                                                          options=options)
                if isinstance(property_position, tuple):
                    property_file = property_position[0]
                    if property_file == '':
                        pass
                    else:
                        with open(property_file, 'rb') as qr:
                            self.stg3_extracted_origin_property = pickle.load(qr)
                        self.infobox.setText(property_file)
                        origin_property_status = 1
                else:
                    pass
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            origin_property_status = 1

        print(int(self.stg4_searching_times_combo_box.currentText()))
        if pca_status == 1 and ellipse_statical_status == 1 and origin_property_status == 1:
            for i in range(len(self.stg3_extracted_origin_property)):
                self.stg4_begin_search_btn.setDisabled(True)
                self.stg4_task_thread.start()
                self.stg4_param.emit(
                    i,
                    self.stg2_pca_model.pca_model,
                    self.stg2_pca_model.preprocessor,
                    self.stg3_ellipse_statical_df_list[i],
                    self.stg3_extracted_origin_property['origin_property_df_list'][i],
                    self.stg3_extracted_origin_property['origin_property_header_list'],
                    int(self.stg4_searching_times_combo_box.currentText()),
                    self.stg2_pca_model.scores_df.shape[1]
                )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec())
