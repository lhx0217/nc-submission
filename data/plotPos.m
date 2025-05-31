function plotPos()
    % 加载数据
    pos_km = loadjson('./run_data/pos_km_1.json');
    pos_ms = loadjson('./run_data/pos_ms_1.json');
    pos_heal = loadjson('./run_data/heal_pos_1.json');
    pos_sci = loadjson('./run_data/sci_pos_1.json');
    % 定义颜色
    startColor = [0.35 0.35 0.35]; % 淡灰色
    endColor = [0 0.4470 0.7410]; % #0072BD
    alphaValue = 0.12; % 透明度
    markersize = 30;
    % 定义立方体的边长
    edgeLength_g = 1;
    graph = loadjson('mouse.json');
    graph(:,2) = -graph(:,2);
    figure('Units','pixels','Position',[100 400 2000 400],'Color','w'); % 总宽1600px，高400px
    % 绘制 pos_km 的轨迹,
    subplot(1, 4, 1);
    hold on;
    view(2);
    xlim([-50 50]);
    ylim([-50 50]);
    for i = 1:size(pos_km, 2)
        point_trajectory = pos_km(1:end, i, 1:2);
        [M, one, N] = size(point_trajectory);
        point_trajectory = reshape(point_trajectory, [M, N]);
        point_trajectory(:, 2) = -point_trajectory(:, 2);
        point_trajectory2 = [point_trajectory; [nan, nan]];
        % 生成与轨迹点数量相等的透明度值数组
        alpha_values = ones(size(point_trajectory2, 1), 1) * alphaValue;
        % c=colorcube(size(point_trajectory2, 1));
        c = lines(size(pos_km, 2));
        % 使用patch绘制轨迹
        patch(point_trajectory2(:, 1), point_trajectory2(:, 2), 'green', ...
        'EdgeColor', c(i,:), ...
        'FaceVertexAlphaData', alpha_values, ...
        'AlphaDataMapping', 'none', ...
        'EdgeAlpha', 'interp', ...
        'FaceAlpha', 'interp',...
        'LineWidth', 2);
        
        % 绘制起始点
        scatter(point_trajectory(1, 1), point_trajectory(1, 2), 20, startColor, 'filled');
        
        % 绘制终点
        scatter(point_trajectory(end, 1), point_trajectory(end, 2), markersize, endColor, 'filled');
    end

    for i = 1:size(graph, 1)
        % 获取当前点的坐标（二维坐标，忽略z值）
        x = graph(i, 1);
        y = graph(i, 2);
        
        % 定义二维方格的4个顶点（基于中心点坐标）
        edgeLength_g_half = edgeLength_g / 2; % 计算边长的一半
        vertices = [
            x - edgeLength_g_half, y - edgeLength_g_half; % 左下点
            x + edgeLength_g_half, y - edgeLength_g_half; % 右下点
            x + edgeLength_g_half, y + edgeLength_g_half; % 右上点
            x - edgeLength_g_half, y + edgeLength_g_half  % 左上点
        ];
        
        % 定义面的连接顺序（二维只需一个面）
        faces = [1 2 3 4];
        
        % 绘制二维方格
        patch('Vertices', vertices,...
              'Faces', faces,...
              'FaceColor', 'g',...
              'EdgeColor', 'none',...
              'FaceAlpha', 0.1);
    end
    %% 图形美化
    ax = gca;
    ax.FontName = 'Arial';
    ax.FontSize = 18;
    ax.YLabel.String = 'X (m)';
    ax.XLabel.String = 'Y (m)';
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.5;
    grid off;
    box on;
    axis equal;

    % 绘制 pos_ms 的轨迹
    subplot(1, 4, 2);
    hold on;
    view(2);
    xlim([-50 50]);
    ylim([-50 50]);
    for i = 1:size(pos_ms, 2)
        point_trajectory = pos_ms(:, i, 1:2);
        [M, one, N] = size(point_trajectory);
        point_trajectory = reshape(point_trajectory, [M, N]);
        point_trajectory(:, 2) = -point_trajectory(:, 2);
        point_trajectory2 = [point_trajectory; [nan, nan]];
        % 生成与轨迹点数量相等的透明度值数组
        alpha_values = ones(size(point_trajectory2, 1), 1) * alphaValue;
        % c=colorcube(size(point_trajectory2, 1));
        c = lines(size(pos_km, 2));
        % 使用patch绘制轨迹
        patch(point_trajectory2(:, 1), point_trajectory2(:, 2), 'green', ...
        'EdgeColor', c(i,:), ...
        'FaceVertexAlphaData', alpha_values, ...
        'AlphaDataMapping', 'none', ...
        'EdgeAlpha', 'interp', ...
        'FaceAlpha', 'interp',...
        'LineWidth', 2);
        
        % 绘制起始点
        scatter(point_trajectory(1, 1), point_trajectory(1, 2), 20, startColor, 'filled');
        
        % 绘制终点
        scatter(point_trajectory(end, 1), point_trajectory(end, 2), markersize, endColor, 'filled');
    end
    set(findobj(gca, 'Type', 'scatter'), 'MarkerFaceAlpha', 0.8);
    % 循环遍历graph中的每个点
    for i = 1:size(graph, 1)
        % 获取当前点的坐标（二维坐标，忽略z值）
        x = graph(i, 1);
        y = graph(i, 2);
        
        % 定义二维方格的4个顶点（基于中心点坐标）
        edgeLength_g_half = edgeLength_g / 2; % 计算边长的一半
        vertices = [
            x - edgeLength_g_half, y - edgeLength_g_half; % 左下点
            x + edgeLength_g_half, y - edgeLength_g_half; % 右下点
            x + edgeLength_g_half, y + edgeLength_g_half; % 右上点
            x - edgeLength_g_half, y + edgeLength_g_half  % 左上点
        ];
        
        % 定义面的连接顺序（二维只需一个面）
        faces = [1 2 3 4];
        
        % 绘制二维方格
        patch('Vertices', vertices,...
              'Faces', faces,...
              'FaceColor', 'g',...
              'EdgeColor', 'none',...
              'FaceAlpha', 0.1);
    end

    %% 图形美化
    ax = gca;
    ax.FontName = 'Arial';
    ax.FontSize = 18;
    ax.YLabel.String = 'X (m)';
    ax.XLabel.String = 'Y (m)';
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.5;
    grid off;
    box on;
    axis equal;

    subplot(1, 4, 3);
    hold on;
    view(2);
    xlim([-50 50]);
    ylim([-50 50]);
    for i = 1:size(pos_heal, 2)
        point_trajectory = pos_heal(:, i, :);
        [M, one, N] = size(point_trajectory);
        point_trajectory = reshape(point_trajectory, [M, N]);
        point_trajectory(:, 2) = -point_trajectory(:, 2);
        point_trajectory2 = [point_trajectory; [nan, nan]];
        % 生成与轨迹点数量相等的透明度值数组
        alpha_values = ones(size(point_trajectory2, 1), 1) * alphaValue;
        % c=colorcube(size(point_trajectory2, 1));
        c = lines(size(pos_km, 2));
        % 使用patch绘制轨迹
        patch(point_trajectory2(:, 1), point_trajectory2(:, 2), 'green', ...
        'EdgeColor', c(i,:), ...
        'FaceVertexAlphaData', alpha_values, ...
        'AlphaDataMapping', 'none', ...
        'EdgeAlpha', 'interp', ...
        'FaceAlpha', 'interp',...
        'LineWidth', 2);
        
        % 绘制起始点
        scatter(point_trajectory(1, 1), point_trajectory(1, 2), 20, startColor, 'filled');
        
        % 绘制终点
        scatter(point_trajectory(end, 1), point_trajectory(end, 2), markersize, endColor, 'filled');
    end

    for i = 1:size(graph, 1)
    % 获取当前点的坐标（二维坐标，忽略z值）
    x = graph(i, 1);
    y = graph(i, 2);
    
    % 定义二维方格的4个顶点（基于中心点坐标）
    edgeLength_g_half = edgeLength_g / 2; % 计算边长的一半
    vertices = [
        x - edgeLength_g_half, y - edgeLength_g_half; % 左下点
        x + edgeLength_g_half, y - edgeLength_g_half; % 右下点
        x + edgeLength_g_half, y + edgeLength_g_half; % 右上点
        x - edgeLength_g_half, y + edgeLength_g_half  % 左上点
    ];
    
    % 定义面的连接顺序（二维只需一个面）
    faces = [1 2 3 4];
    
    % 绘制二维方格
    patch('Vertices', vertices,...
          'Faces', faces,...
          'FaceColor', 'g',...
          'EdgeColor', 'none',...
          'FaceAlpha', 0.1);
    end
    %% 图形美化
    ax = gca;
    ax.FontName = 'Arial';
    ax.FontSize = 18;
    ax.YLabel.String = 'X (m)';
    ax.XLabel.String = 'Y (m)';
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.5;
    grid off;
    box on;
    axis equal;

    subplot(1, 4, 4);
    hold on;
    view(2);

    t = pos_sci(:, :,1);
    pos_sci(:,:, 1) = pos_sci(:,:, 2);
    pos_sci(:, :,2) = t;
    for i = 1:size(pos_sci, 2)
        point_trajectory = pos_sci(:, i, :);
        [M, one, N] = size(point_trajectory);
        point_trajectory = reshape(point_trajectory, [M, N]);
        point_trajectory(:, 2) = -point_trajectory(:, 2);
        point_trajectory2 = [point_trajectory; [nan, nan]];
        % 生成与轨迹点数量相等的透明度值数组
        alpha_values = ones(size(point_trajectory2, 1), 1) * alphaValue;
        % c=colorcube(size(point_trajectory2, 1));
        c = lines(size(pos_km, 2));
        % 使用patch绘制轨迹
        patch(point_trajectory2(:, 1), point_trajectory2(:, 2), 'green', ...
        'EdgeColor', c(i,:), ...
        'FaceVertexAlphaData', alpha_values, ...
        'AlphaDataMapping', 'none', ...
        'EdgeAlpha', 'interp', ...
        'FaceAlpha', 'interp',...
        'LineWidth', 2);
        
        % 绘制起始点
        scatter(point_trajectory(1, 1), point_trajectory(1, 2), 20, startColor, 'filled');
        
        % 绘制终点
        scatter(point_trajectory(end, 1), point_trajectory(end, 2), 40, endColor, 'filled');
    end

    for i = 1:size(graph, 1)
    % 获取当前点的坐标（二维坐标，忽略z值）
    x = graph(i, 1);
    y = graph(i, 2);
    
    % 定义二维方格的4个顶点（基于中心点坐标）
    edgeLength_g_half = edgeLength_g / 2; % 计算边长的一半
    vertices = [
        x - edgeLength_g_half, y - edgeLength_g_half; % 左下点
        x + edgeLength_g_half, y - edgeLength_g_half; % 右下点
        x + edgeLength_g_half, y + edgeLength_g_half; % 右上点
        x - edgeLength_g_half, y + edgeLength_g_half  % 左上点
    ];
    
    % 定义面的连接顺序（二维只需一个面）
    faces = [1 2 3 4];
    
    % 绘制二维方格
    patch('Vertices', vertices,...
          'Faces', faces,...
          'FaceColor', 'g',...
          'EdgeColor', 'none',...
          'FaceAlpha', 0.1);
    end
    %% 图形美化
    ax = gca;
    ax.FontName = 'Arial';
    ax.FontSize = 18;
    ax.YLabel.String = 'X (m)';
    ax.XLabel.String = 'Y (m)';
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.5;
    xlim([-50 50]);
    ylim([-50 50]);
    ax.XTick = -50:50:50;
    ax.YTick = -50:50:50;
    grid off;
    box on;
    axis equal;

end