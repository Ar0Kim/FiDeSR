
python test_fidesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-1-base \
--pretrained_path preset/models/fidesr.pkl \
--process_size 512 \
--upscale 4 \
--input_image preset/test_datasets \
--output_dir experiments/test \
--hf_scale 0.3 \
--lf_scale 0
