# eps with percent
dataset=(subj)
# eps=(1 0.1 0.5 0.01 0.05 0.001 0.005 0.0001)
eps=(0.1 0.5 0.01 0.001 0.005 0.0001 1)
for data in ${dataset[@]};do
    for epsilon in ${eps[@]};do
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 4 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 4 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 4 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 4 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
    done
done

dataset=(mpqa subj)
# eps=(1 0.1 0.5 0.01 0.05 0.001 0.005 0.0001)
eps=(0.1 0.5 0.01 0.001 0.005 0.0001 1)
for data in ${dataset[@]};do
    for epsilon in ${eps[@]};do
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data -train_on_base_model_label
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 4 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data -train_on_base_model_label
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 4 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data -train_on_base_model_label
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data -train_on_base_model_label
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 4 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data -train_on_base_model_label
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 4 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data -train_on_base_model_label
    done
done


# eps with percent
# eps=(1 0.1 0.5 0.01 0.05 0.001 0.005 0.0001)
eps=(0.1 100)
dataset=(mpqa subj)
for data in ${dataset[@]};do
    for epsilon in ${eps[@]};do
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 4 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type cnn -snapshot saved_model/best_cnn.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 4 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 4 1 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
        CUDA_VISIBLE_DEVICES=0 python main.py -dataset $data -model_type lstm -snapshot saved_model/best_lstm.pt -adv_train -model_eps_ball -eps $epsilon -wgt 1 4 0 -epochs 10  -optim_algo adam -lr 0.001 -add_pos_adv_data
    done
done

