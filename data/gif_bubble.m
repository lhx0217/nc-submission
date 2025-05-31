function plotPos3D_animated()
    % 加载轨迹数据
    pos_km = loadjson('./run_data/pos_km_mountain.json');
    graph = loadjson('graph.json');
    graph(:, 2) = -graph(:, 2);  % Y轴翻转

    % 可视化参数
    startColor = [0.35 0.35 0.35];
    endColor = [0 0.4470 0.7410];
    trajAlpha = 0.15; % 轨迹透明度
    currentMarkerSize = 20;
    currentMarkerColor = [0 0.4470 0.7410];

    % 初始化
    figure('Units','pixels','Position',[100 100 600 600],'Color','w');
    hold on; view(2); axis equal;
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    grid on; box on;
    xlim([-35 35]); ylim([-35 35]); zlim([-35 35]);

    % graph 点
    scatter3(graph(:,1), graph(:,2), graph(:,3), ...
             30, 'g', 'filled', 'MarkerFaceAlpha', 0.08);

    % 轨迹数据
    numAgents = size(pos_km, 2);
    numSteps = size(pos_km, 1);
    l = lines(numAgents);
    h_lines = gobjects(numAgents, 1); % 每条轨迹的 handle
    h_robot_circles = gobjects(numAgents, 1);
    theta = linspace(0, 2*pi, 50);  % 圆的角度分布
    r = 1;  % 半径为1米
    traj_data = cell(numAgents, 1);
    start_pts = zeros(numAgents, 3);
    end_pts = zeros(numAgents, 3);

    for i = 1:numAgents
        traj_data{i} = nan(numSteps, 3);
        start_pts(i, :) = reshape(pos_km(1, i, :), [1, 3]);
        end_pt = reshape(pos_km(end, i, :), [1, 3]);
        end_pt(2) = -end_pt(2);
        end_pts(i, :) = end_pt;

        % 初始化轨迹线（透明颜色）
        c = l(i,:) * (1 - trajAlpha) + [1 1 1] * trajAlpha; % 调整亮度
        h_lines(i) = patch('XData', NaN, 'YData', NaN, 'ZData', NaN, ... 
                           'EdgeColor', l(i,:), ...
                           'EdgeAlpha', 0.2, ...
                           'LineWidth', 1.5, ...
                           'FaceColor', 'none');  % 不填充面
    end

    % 起点
    scatter3(start_pts(:,1), -start_pts(:,2), start_pts(:,3), ...
             10, repmat(startColor, numAgents,1), 'filled');


    % 保存为 GIF
    gifPath = '/home/lhx/Desktop/trans7/km_mountain.gif';
    if exist(gifPath, 'file'), delete(gifPath); end

    for t = 1:numSteps
        pos_now = zeros(numAgents, 3);

        for i = 1:numAgents
            pt = reshape(pos_km(t, i, :), [1, 3]);
            pt(2) = -pt(2);  % 翻转 Y 轴
            traj_data{i}(t, :) = pt;
            data = traj_data{i}(1:t, :);
            data = [data; [NaN, NaN, NaN]];
            % 更新轨迹线
            set(h_lines(i), 'XData', data(:, 1), ...
                            'YData', data(:, 2), ...
                            'ZData', data(:, 3));

            % 当前位置信息
            pos_now(i, :) = pt;
            if isgraphics(h_robot_circles(i))
                delete(h_robot_circles(i));
            end
        
            % 生成圆边界坐标（XY平面上）
            x_c = r * cos(theta) + pos_now(i,1);
            y_c = r * sin(theta) + pos_now(i,2);
            z_c = ones(size(x_c)) * pos_now(i,3);  % 圆位于当前 z 高度
        
            % 绘制当前机器人圆
            h_robot_circles(i) = fill3(x_c, y_c, z_c, currentMarkerColor, ...
                                        'FaceAlpha', 0.2, ...
                                        'EdgeColor', 'k', ...
                                        'LineWidth', 1.2);
        end

        

        drawnow;

        % 生成帧并写入 GIF
        frame = getframe(gcf);
        im = frame2im(frame);
        [A, map] = rgb2ind(im, 256);
        if t == 1
            imwrite(A, map, gifPath, 'gif', 'LoopCount', Inf, 'DelayTime', 0.05);
        else
            imwrite(A, map, gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
        end
    end

    % 终点
    viewpoint = campos;
    distances = vecnorm(end_pts - viewpoint, 2, 2);
    markerSizes = rescale(max(distances) - distances, 35, 35);
    scatter3(end_pts(:,1), end_pts(:,2), end_pts(:,3), ...
             markerSizes, repmat(endColor, numAgents,1), 'filled');

    fprintf('GIF 动图已保存至: %s\n', gifPath);
end
