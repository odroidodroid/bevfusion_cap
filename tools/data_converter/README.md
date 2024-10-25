<h1 id="create-reduced-dataset">Create Reduced Dataset</h1>
<h3 id="run-command-reduce-ratio--0.1">Run command (reduce ratio = 0.1)</h3>
<p>modify reduce-ratio that you want</p>
<pre><code> $ python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --reduce-ratio=0.1
</code></pre>
<h3 id="after-that-you-can-get">After that, you can get</h3>
<pre><code>$ ls
nuscenes_reduced0.1_dbinfos_train.pkl
nuscenes_reduced0.1_gt_database
nuscenes_reduced0.1_infos_test.pkl
nuscenes_reduced0.1_infos_train.pkl
nuscenes_reduced0.1_infos_val.pkl
</code></pre>
<h3 id="train-bevfusion-with-reduced-dataset">Train BEVFusion with reduced dataset</h3>
<pre><code>$ torchpack dist-run -np 1 python tools/train.py \
configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
--load_from pretrained/swint-nuimages-pretrained.pth \
--data.train.dataset.reduce_ratio=0.1 \
--data.val.reduce_ratio=0.1 \
--data.train.dataset.ann_file=data/nuscenes/nuscenes_reduced0.1_infos_train.pkl \
--data.val.ann_file=data/nuscenes/nuscenes_reduced0.1_infos_val.pkl
</code></pre>
<h3 id="test-bevfusion-with-reduced-dataset">Test BEVFusion with reduced dataset</h3>
<pre><code>$ torchpack dist-run -np 1 python tools/test.py \
configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
pretrained/bevfusion-det.pth \
--data.test.reduce_ratio=0.1 \
--data.test.ann_file=data/nuscenes/nuscenes_reduced0.1_infos_val.pkl \
--eval bbox
</code></pre>
<h3 id="tips-for-configuring-lr-and-warmup-iterations">Tips for configuring lr and warmup iterations</h3>
<img width="515" alt="lr_config_tips" src="https://github.com/user-attachments/assets/e563a98d-cb4f-423f-9f9d-a4d197c3c283">

<img width="1449" alt="reduced_dataset_table" src="https://github.com/user-attachments/assets/c1f1681f-f4f9-4321-ab51-7c4899230678">
