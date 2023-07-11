import tensorrt as trt
import segment_anything.calib.calib_dataloader as calib_dataloader
import os

class SAMCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        trt.IInt8EntropyCalibrator2.__init__(self)
        device = 'cuda:0'
        orig_image_shape=[1080, 1920]
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        vit_calib_dir='%s/calib_data/vit' % cur_dir
        self.dataloader = calib_dataloader.CalibDataloader(device=device, origin_image_shape=orig_image_shape, vit_calib_dir=vit_calib_dir)
        self.batch_size = 1
        self.cache_file = "%s/int8calib.cache" % cur_dir
        
    def get_batch_size(self):
        return self.batch_size
        
    def get_batch(self, names):
        try:
            result = []
            ort_inputs = next(self.dataloader)
            for name in names:
                result.append(ort_inputs[name].data_ptr())
            return result
        except StopIteration:
            return None
        
    def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
            
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

if __name__ == '__main__':
    calib = SAMCalibrator()
    print(calib.read_calibration_cache())