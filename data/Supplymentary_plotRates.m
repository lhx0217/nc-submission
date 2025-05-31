function plotRates()
    % 文件名
    files = {'./run_data/rate_ms_tree1.json'};
    numFiles = length(files);

    % 初始化数据存储
    metrics = {'entering_rates', 'covering_rates', 'uniform_rates', ...
               'move', 'avg_des', 'mean_vel', 'std_dist2', 'std_contain', ...
               'min_dist', 'times'};
    for m = 1:length(metrics)
        eval([metrics{m} ' = cell(numFiles, 1);']);
    end

    % 读取数据
    for i = 1:numFiles
        data = loadjson(files{i});
        entering_rates{i} = data(:, 1)';
        covering_rates{i} = data(:, 2)';
        uniform_rates{i} = data(:, 3)';
        move{i} = data(:, 4)';
        avg_des{i} = data(:, 5)';
        mean_vel{i} = data(:, 6)';
        std_dist2{i} = data(:, 7)';
        std_contain{i} = data(:, 8)';
        min_dist{i} = data(:, 9)';
        times{i} = data(:, 10)' * 0.23;
    end

    % 要绘图的指标及标签
    metric_list = {
        'covering_rates', 'Coverage Rate';
        'entering_rates', 'Entering Rate';
        'uniform_rates', 'Uniformity';
        'min_dist', 'Min Distance (m)';
    };

    % 配色列表（主色，阴影色）
    color_main_list = {
        [0.2, 0.4, 0.8];   % 深蓝
        [0.85, 0.33, 0.1]; % 橙红
        [0.2, 0.6, 0.4];   % 墨绿
        [0.5, 0.2, 0.5];   % 紫红
    };
    color_fill_list = {
        [0.6, 0.7, 0.95];  % 浅蓝
        [1.0, 0.7, 0.5];   % 浅橙
        [0.6, 0.9, 0.7];   % 浅绿
        [0.85, 0.7, 0.9];  % 粉紫
    };

    for k = 1:size(metric_list, 1)
        metric_name = metric_list{k, 1};
        y_label = metric_list{k, 2};
        color_main = color_main_list{k};
        color_fill = color_fill_list{k};

        % 获取数据矩阵
        raw_data = eval(metric_name);
        t = times{1}; % 统一时间轴

        % 对齐时间，插值
        aligned_data = zeros(length(t), numFiles);
        for i = 1:numFiles
            aligned_data(:, i) = interp1(times{i}, raw_data{i}, t, 'linear', 'extrap');
        end

        % 计算统计量
        mean_vals = mean(aligned_data, 2);
        min_vals = min(aligned_data, [], 2);
        max_vals = max(aligned_data, [], 2);

        % 创建图形
        figure('Position', [100 100 750 600]);
        hold on;

        % 阴影区
        fill([t, fliplr(t)], [max_vals', fliplr(min_vals')], ...
            color_fill, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

        % 平均曲线
        plot(t, mean_vals, '-', 'Color', color_main, 'LineWidth', 3);

        % min_dist 特殊标记
        if strcmp(metric_name, 'min_dist')
            yline_val = 0.4;
            plot(t, yline_val * ones(size(t)), '--', 'Color', [0 0.6 0], 'LineWidth', 2);
        end

        % 图形美化
        grid on;
        box off;
        set(gca, 'LineWidth', 2.5);
        set(gca, 'FontSize', 24);
        xlabel('Time (s)', 'FontSize', 26);
        ylabel(y_label, 'FontSize', 26);
        xlim([0 460])
        legend({'Max-Min Range', 'Mean'}, 'Location', 'best', 'FontSize', 18);
        ax = gca;
        ax.FontName = 'Arial';
        ax.FontWeight = 'bold';
    end
end
