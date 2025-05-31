function plotPos3D()
    % 加载轨迹数据
    pos_km = loadjson('./run_data/pos_km_tree3.json');

    % 加载 graph 数据
    graph = loadjson('graph.json');
    graph(:, 2) = -graph(:, 2);  % Y轴翻转

    % 可视化参数
    startColor = [0.35 0.35 0.35];
    endColor = [0 0.4470 0.7410];
    alphaValue = 0.12;
    markersize = 30;
    edgeLength_g = 1;  % 立方体边长
    % 初始化起点终点坐标和颜色数组    
    figure('Units','pixels','Position',[100 100 600 600],'Color','w');
    hold on;
    view(2);
    axis equal;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    l = lines(size(pos_km, 2));
    end_pts = [];
    start_pts = [];
    % 绘制轨迹
    for i = 1:size(pos_km, 2)
        point_trajectory = pos_km(1:end, i, :);
        [M, one, N] = size(point_trajectory);
        point_trajectory = reshape(point_trajectory, [M, N]);
        point_trajectory(:, 2) = -point_trajectory(:, 2);  % 翻转 Y
        point_trajectory = [point_trajectory; [nan, nan, nan]];
        % 绘制轨迹线（使用 plot3）
        patch('XData', point_trajectory(:, 1), ...
              'YData', point_trajectory(:, 2), ...
              'ZData', point_trajectory(:, 3), ...
              'EdgeColor', l(i,:), ...
              'EdgeAlpha', 0.1, ...       % 设置透明度
              'LineWidth', 0.5, ...
              'FaceColor', 'none');       % 不填充面
        start_pts = [start_pts; point_trajectory(1, :)];
        end_pt = point_trajectory(end-1, :);
        end_pts = [end_pts; end_pt];
    end
    viewpoint = campos;
    
    % 计算每个点到相机的距离
    distances = vecnorm(end_pts - viewpoint, 2, 2);  % 欧氏距离
    
    % 将距离转为 marker 大小（距离越小，点越大）
    markerSizes = rescale(max(distances) - distances, 35, 35);  % 可调大小范围
    % 批量绘制起点
    scatter3(start_pts(:, 1), start_pts(:, 2), start_pts(:, 3), ...
             5, repmat(startColor, size(start_pts,1), 1), 'filled');
    
    % 批量绘制终点（使用动态大小）
    scatter3(end_pts(:, 1), end_pts(:, 2), end_pts(:, 3), ...
             markerSizes, repmat(endColor, size(end_pts,1), 1), 'filled');

    % 绘制 3D 立方体（graph 中每个点一个立方体）
    % 获取当前相机位置（视点位置）
    viewpoint = campos;
    
    % 计算每个点到相机的距离
    distances = vecnorm(graph - viewpoint, 2, 2);  % 欧氏距离
    
    % 将距离转为 marker 大小（距离越小，点越大）
    markerSizes = rescale(max(distances) - distances, 20, 60);  % 可调大小范围
    
    % 绘制可变大小的 scatter3 点
    scatter3(graph(:,1), graph(:,2), graph(:,3), ...
             markerSizes, 'g', 'filled', ...
             'MarkerFaceAlpha', 0.08);
    % 图形美化
    ax = gca;
    ax.FontName = 'Arial';
    ax.FontSize = 18;
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.5;
    grid on;
    box on;
    xlim([-35 35]);
    ylim([-35 35]);
    zlim([-35 35]);
    % 确保目录存在
    saveDir = '/home/lhx/Desktop/trans7';
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    
    % 保存为PNG（600 DPI）
    savePath = fullfile(saveDir, 'km_tree.png');
    print(gcf, savePath, '-dpng', '-r600');
    % 或者保存为PDF（矢量图，更清晰）
    % savePath = fullfile(saveDir, '3D_trajectory.pdf');
    % exportgraphics(gcf, savePath, 'ContentType', 'vector', 'Resolution', 600);
    
    fprintf('图像已保存至: %s\n', savePath);
end
