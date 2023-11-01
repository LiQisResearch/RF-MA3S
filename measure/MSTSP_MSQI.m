function [MSQI,DIFF,OPTI] = MSTSP_MSQI(alg_solution, index, theta,best_route)

if isempty(alg_solution)
    MSQI=0;
    return
elseif index < 1 || index > 25
    fprintf('The index is out of range(1-25).\n');
    return
end

    city_num = size(alg_solution, 2) - 1;
    size_solution=size(alg_solution, 1);
if size(alg_solution,1)>1
    share_dist = zeros(size(alg_solution, 1), size(alg_solution, 1));
    for i = 1:size(alg_solution, 1)
        for j = 1: size(alg_solution, 1)
            share_dist(i, j) = measure_share_dist(alg_solution(i, 2:end), alg_solution(j, 2:end));
        end
    end
else
    share_dist=city_num;
end


if size(alg_solution,1)>1
    difference_dist=city_num-share_dist;
    difference_mindist=2*sum(difference_dist,2)/(city_num*(size(alg_solution,1)-1));
    difference_mindist(difference_mindist>1)=1;
else
    difference_mindist=0;
end


opti=((theta+1)*best_route-alg_solution(:,1))/(theta*best_route);

DIFF=mean(difference_mindist);
OPTI=mean(opti);
SQI =2./(1./opti+1./difference_mindist);
MSQI=size_solution/(sum(1./SQI));


end