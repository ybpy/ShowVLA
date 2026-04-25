# (20, -140, 10, -10) for stack_boxes
# (180, -60, 0, -300) for JAKA_battery_two, JAKA_open_bottle, clutter
# (40, -140, 300, -300) for other tasks

# python convert_jaka.py \
#     --data_dir /home/hyx/JAKA_data/USB_light_plug \
#     --output_dir /home/hyx/datasets/JAKA/USB_light_plug \
#     --meta_prefix JAKA \
#     --speed_up 1 \
#     --image_stream_offset 1 \
#     --crop_main "(40, -140, 300, -300)" \
#     --overwrite

python convert_jaka.py \
    --data_dir /home/hyx/JAKA_data/clutter_put \
    --output_dir /home/hyx/datasets/JAKA/clutter_put \
    --meta_prefix JAKA \
    --speed_up 1 \
    --image_stream_offset 1 \
    --crop_main "(180, -60, 0, -300)"

python convert_jaka.py \
    --data_dir /home/hyx/JAKA_data/clutter_put2 \
    --output_dir /home/hyx/datasets/JAKA/clutter_put2 \
    --meta_prefix JAKA \
    --speed_up 1 \
    --image_stream_offset 1 \
    --crop_main "(180, -60, 0, -300)"