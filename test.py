from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util_nifti
import os

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    predictions = []
    
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # number of slices = higher than default value of --num_test (50)
        #     break
        
        model.set_input(data)
        model.test()
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        predictions.append({
            'visuals': visuals,  # Pass full visuals dictionary
            'path': img_path[0],
            'slice_start': data['slice_start'].item()
        })

        # if i % 5 == 0:
        #     print(f'Processing {i:05d}-th sliding window... / {os.path.basename(img_path[0])}...')
    
    save_dir = os.path.join(opt.results_dir, opt.name, 'test_latest')
    util_nifti.process_and_save_predictions(
        predictions,
        patient_original_path=img_path[0],
        save_dir=save_dir
    )