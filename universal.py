from fastai.text import *
import pdb
# -------------------------------------------------------------------------------------------
class ULMFiT(object):
# -------------------------------------------------------------------------------------------

    def __init__(self, max_lr, lrn_type='lin'):
        LRN_TYPES = {
            'lin': [max_lr/4, max_lr/2, max_lr, max_lr/2], 
            'exp': [max_lr/4, max_lr, max_lr*4, max_lr],
            'osc': [max_lr/4, max_lr, max_lr/4, max_lr]
        }
        self.max_lr = max_lr
        self.lrn_type = lrn_type
        self.lrn_rates = LRN_TYPES[lrn_type]
        return
    
    @property
    def data_clas(self):
        return self._data_clas
    
    @data_clas.setter
    def data_clas(self, data_clas):
        self._data_clas = data_clas
    
    @property
    def lm_data(self):
        return self._lm_data
    
    @lm_data.setter
    def lm_data(self, lm_data):
        self._lm_data = lm_data
        
    @property
    def lrn_rates(self):
        return self._lrn_rates
    
    @lrn_rates.setter
    def lrn_rates(self, lrn_rates):
        self._lrn_rates = lrn_rates
    
    @property
    def max_lr(self):
        return self._max_lr
    
    @max_lr.setter
    def max_lr(self, max_lr):
        self._max_lr = max_lr
        

    # ---------------------------------------------------------------------------------------
    # I. Finetune Pretrained Language Model
    # ---------------------------------------------------------------------------------------
    def finetune(self, label_cols, text_cols, trn_data, val_data, cuda_id=None, cyc_len=1, 
                    batch_sz=32, drop_mult=0.5, wds=[1e-7], max_lr=1e-2, lrn_type='lin', 
                    train_id='air-tweets'):
        """ 
            ---------------------------------------------------------------------------------
            Args:
            
                label_cols (str): name of label columns in dataset
                text_cols (str): name of text columns in dataset
                trn_data (pandas.DataFrame): training dataset
                val_data (pandas.DataFrame): validation dataset
                cuda_id (str): id to set up cuda cores
                cyc_len (int): cycle length of training and classification
                batch_sz (int): batch size of input data
                drop_mult (float): dropout multiplier for learning model
                wds (arr): list of weight decays
                max_lr (float): max learning rate
                lrn_type (str): 
            ---------------------------------------------------------------------------------
        """ 
        if not self.max_lr:
            self.max_lr = max_lr 
        # ------------------------------------------------------------------------------------
        # setup cuda cores if available
        # ------------------------------------------------------------------------------------
        if cuda_id:
            torch.cuda.set_device(cuda_id); print(f'Connecting to... cuda_id={cuda_id}')
        else:
            print('CUDA not avail')
        # ------------------------------------------------------------------------------------
        # setup data objs
        # ------------------------------------------------------------------------------------
        self.lm_data = TextLMDataBunch.from_df(path='', train_df=trn_data, valid_df=val_data,
                                               label_cols=label_cols, text_cols=text_cols)
        self.clas_data = TextClasDataBunch.from_df(path='', train_df=trn_data, valid_df=val_data,
                                                   label_cols=label_cols, text_cols=text_cols,
                                                   vocab=self.lm_data.train_ds.vocab, bs=batch_sz)
        learner = language_model_learner(self.lm_data, arch=AWD_LSTM, drop_mult=drop_mult, 
                                         pretrained=True)
        # ------------------------------------------------------------------------------------
        # fit, save, record losses
        # ------------------------------------------------------------------------------------
        learner.fit(cyc_len, lr=self.lrn_rates, wd=wds)
        learner.save_encoder(f'{train_id}_enc')
        learner.recorder.plot_losses()
        plt.grid()
        plt.title(f'Loss vs. Epoch\nlrs={self.lrn_rates}, wd={wds}')
#         learn = text_classifier_learner(self.clas_data, drop_mult=drop_mult, arch=AWD_LSTM)
#         learn.load_encoder(f'{train_id}_enc')
#         learn.fit_one_cycle(cyc_len, self.max_lr)
#         learn.save(f'trnd_{train_id}_enc')
        return
        
    # ---------------------------------------------------------------------------------------
    # II. Classify & Predict
    # ---------------------------------------------------------------------------------------     
    def classify(self, cyc_len=1, cuda_id=None, train_id='air-tweets', drop_mult=0.5):
        # ------------------------------------------------------------------------------------
        # setup cuda cores if available
        # ------------------------------------------------------------------------------------
        if not hasattr(torch._C, '_cuda_setDevice'):
            print('CUDA not available. Setting device=-1.')
            cuda_id = -1
        torch.cuda.set_device(cuda_id) if cuda_id else print('CUDA Not Avail...')
        # ------------------------------------------------------------------------------------
        # setup learner obj, load encoder, fit
        # ------------------------------------------------------------------------------------
        learner = text_classifier_learner(self.clas_data, drop_mult=drop_mult, arch=AWD_LSTM)
        learner.load_encoder(f'{train_id}_enc')
        learner.fit_one_cycle(cyc_len, self.max_lr)
        learner.save(f'trnd_{train_id}_enc')
        learner.recorder.plot_losses()
        plt.grid()
        plt.title(f'Loss vs. Epoch')
        return learner