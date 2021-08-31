id_start=100
id_len=5
for ((pred_id=${id_start};pred_id<${id_start}+${id_len};pred_id++))
do
sbatch noCls_gco.sh ${pred_id}
done
