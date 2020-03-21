python3 bert_pytorch/__main__.py \
	--dataset="/projects/bdata/jupyter/gezhang_backup/jupyter-notebook-analysis/graphs/cell_with_func_python23_1_27.txt" \
	--output_path='temp_7.model' \
	--test_path='/projects/bdata/jupyter/gezhang_backup/jupyter-notebook-analysis/graphs/test.txt' \
	--vocab_path='./temp_7/vocab.txt' \
	--cuda_devices='1' \
	--log_freq=10000 \
	--epochs=50 \
	--layers=4 \
	--attn_heads=4 \
	--lr=0.00003 \
	--batch_size=16 \
	--num_workers=1 \
	--duplicate=1 \
	--dropout=0 \
	--min_occur=1 \
	--weak_supervise \
	--use_sub_token \
	--seq_len=160 \
	--max_graph_num=1000000 \
	--markdown \
	--hinge_loss_start_point=50 \
	--entropy_start_point=50

	# --test_path='/homes/gws/gezhang/jupyter-notebook-analysis/graphs/test_cells_1_27.txt' \
