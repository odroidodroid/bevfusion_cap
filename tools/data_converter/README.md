---


---

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
<pre><code>$ torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
--load_from pretrained/swint-nuimages-pretrained.pth \
--data.train.dataset.reduce_ratio=0.1 \
--data.val.reduce_ratio=0.1 \
--data.train.dataset.ann_file=data/nuscenes/nuscenes_reduced0.1_infos_train.pkl \
--data.val.ann_file=data/nuscenes/nuscenes_reduced0.1_infos_val.pkl
</code></pre>
<h3 id="test-bevfusion-with-reduced-dataset">Test BEVFusion with reduced dataset</h3>
<p>you should modify your configs/nuscenes/default.yaml when you run test in BEVFusion.</p>
<pre><code>data : 
	test:
		type: ${dataset_type}
		dataset_root: ${dataset_root}
		ann_file: ${dataset_root + "nuscenes_reduced0.1_infos_val.pkl"}
		pipeline: ${test_pipeline}
		object_classes: ${object_classes}
		map_classes: ${map_classes}
		modality: ${input_modality}
		test_mode: true
		box_type_3d: LiDAR
		reduce_ratio : 0.1
</code></pre>
<p>and run command</p>
<pre><code>$ torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
</code></pre>
<blockquote>
<p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote>

