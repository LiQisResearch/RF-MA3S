function [Fbeta, DI] = MSTSP_measure(alg_solution, index,mstsp_solution)

beta2 = 0.3;

if isempty(alg_solution)
    Fbeta=0; DI=0;
    return
elseif index < 1 || index > 25
    fprintf('The index is out of range(1-25).\n');
    return
end

mstsp_solution = mstsp_solution(:, 1:end-1);




city_num = size(mstsp_solution, 2) - 1;
share_dist = zeros(size(mstsp_solution, 1), size(alg_solution, 1));
for i = 1:size(mstsp_solution, 1)
    for j = 1: size(alg_solution, 1)
        share_dist(i, j) = measure_share_dist(alg_solution(j, 2:end), mstsp_solution(i, 2:end));
    end
end

flag_mstsp = max(share_dist, [], 2) == repmat(city_num,size(mstsp_solution, 1), 1 );
flag_alg = max(share_dist, [], 1)' == repmat(city_num,size(alg_solution, 1), 1 );
[nearest_num]  = max(share_dist, [], 2);


DI = mean(nearest_num) / city_num;
TP =  sum(flag_alg);
FP =  size(flag_alg, 1) - TP;
FN = size(flag_mstsp, 1) - sum(flag_mstsp);
P = TP / (TP + FP);
R = TP / (TP + FN);
Fbeta = (1+beta2)*P*R /((beta2)*P + R);
if isnan(Fbeta)
    Fbeta = 0;
end

end