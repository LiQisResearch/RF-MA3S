function [alg_solution] = MSTSP_filter(alg_solution, index,theta,shD,best_route)
%shD=0.8;theta=0.1;best_route=mstsp_solution(1,1);


if isempty(alg_solution)
    fprintf('The number of the solutions is empty.\n');
    return
elseif index < 1 || index > 25
    fprintf('The index is out of range(1-25).\n');
    return
end

alg_solution=unique(alg_solution,'row');
alg_solution=alg_solution(find(alg_solution(:,1)<best_route*(1+theta)),:);%Optimality filtering
if isempty(alg_solution)
    return
end
if size(alg_solution,1)<=2
    return
end
city_num = size(alg_solution, 2) - 1;
share_dist = zeros(size(alg_solution, 1), size(alg_solution, 1));

size_solution=size(alg_solution, 1);
sumi=0;
sumj=0;
i=1;
size_solution=sortrows(size_solution,1,'descend');
while 1
    j=i+1;
    while 1
        share_dist(i, j) = measure_share_dist(alg_solution(j, 2:end), alg_solution(i, 2:end));
        if share_dist(i, j)>city_num*shD
            if share_dist(i,j)~=city_num&&(alg_solution(i,1)==best_route&&alg_solution(j,1)==best_route)%If it's optimal, and it's not repeated, then don't clip it
                
            else
                if alg_solution(i,1)<alg_solution(j,1)
                    alg_solution(j,:)=[];
                    share_dist(:, j)=[];share_dist(j,:)=[];
                    size_solution=size(alg_solution, 1);
                    j=j-1;
                    sumj=sumj+1;
                else
                    alg_solution(i,:)=[];
                    share_dist(:, i)=[];share_dist(i,:)=[];
                    size_solution=size(alg_solution, 1);
                    i=i-1;
                    sumi=sumi+1;
                    break;
                end
            end
        end 
        j=j+1;
        if j>size_solution
             break;
        end       
    end
    i=i+1;
    
    if i>=size_solution
        break;
    end
end
alg_solution=sortrows(alg_solution,1,'ascend');

end