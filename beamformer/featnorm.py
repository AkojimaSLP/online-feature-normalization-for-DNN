import numpy as np

class Featnorm:
    def __init__(self, 
                 feature_order=40):        
        self.number_of_frame = 1
        # to Do: initialize by statistics calculated by  development data
        #self.previous_cma = np.zeros(feature_order, dtype=np.float32)
        #self.previous_cms = np.zeros(feature_order, dtype=np.float32)        
        self.previous_cma = np.load('./mean_binary.npy')
        self.previous_cms = np.load('./var_binary.npy')


    def get_current_statistics(self, frame):
        """ frame -> (feature_order, 1)
            return statistics at current frame
        """
        current_cma = self.previous_cma + (frame - self.previous_cma) / (self.number_of_frame + 1)
        current_cms = self.previous_cms + (frame - self.previous_cma) * (frame - current_cma)
        
        self.number_of_frame = self.number_of_frame + 1
        self.previous_cma = current_cma
        self.previous_cms = current_cms
        
        return current_cma, np.sqrt(current_cms / self.number_of_frame)
    
    def get_normalize_frame(self, mean, std, frame):
        """ mean -> (feature_order, 1)
            std -> (feature_order, 1)
            frame -> (feature_order, 1)
        """
        return (frame - mean) / std
