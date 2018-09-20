function [path] = findPathContinuous(reachGrid, goal, start, step)
if nargin < 4
    step = 1;
end
y = goal(1);
x = goal(2);
figure(10)
[px,py] = gradient(reachGrid);
modGrid = sqrt(px.^2 + py.^2);
modGrid(modGrid==0) = 1;
px_plot = -px./modGrid;
py_plot = -py./modGrid;
% contour(reachGrid);
% hold on
% quiver(px_plot,py_plot);
% plot(x, y, 'r*');
% plot(start(1), start(2), 'g*');
path = [x, y];
while true
%     plot(path(:,1), path(:,2), 'r');
    i_high = ceil(y);
    j_high = ceil(x);
    i_low = floor(y);
    j_low = floor(x);
    alpha = j_high - x;
    beta = i_high - y;
    gradX = -[alpha, 1-alpha]*px(i_low:i_high, j_low:j_high)*[beta; 1-beta];
    gradY = -[alpha, 1-alpha]*py(i_low:i_high, j_low:j_high)*[beta; 1-beta];
    if abs(gradX) > 1
        gradX = sign(gradX);
    end
    if abs(gradY) > 1
        gradY = sign(gradY);
    end
    x_next = x + step*gradX;
    y_next = y + step*gradY;
    x = x_next;
    y = y_next;
    path = [x, y; path];
    if norm([x,y]-start) < 0.5
        break
    end
    if size(path,1) >= 1e4
        error('no safe path to frontier')
        break
    end
end
% plot(path(:,1), path(:,2), 'r');
hold off
end
