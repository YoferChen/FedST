# Train
python ./run.py -c config/AAFnew_separate/labeltoimage_AAFnew0.json -p train -gpu 0
# Test
# Set resume_state to the path of trained model in json file
# Set data_client and model_client to generate synthetic images for select data by resume_state
python ./run.py -c config/AAFnew_separate_test/0/AAFnew_data1_style0.json -p train -gpu 0