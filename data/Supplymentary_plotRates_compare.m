function plotRates()
    % 文件名
    files_ms = {'./run_data/rate_ms_tree1.json', './run_data/rate_ms_tree2.json', './run_data/rate_ms_tree3.json'};
    files_km = {'./run_data/rate_km_tree1.json', './run_data/rate_km_tree2.json', './run_data/rate_km_tree3.json'};
    numFiles = length(files_ms);

    % 指标名
    metrics = {'entering_rates', 'covering_rates', 'uniform_rates', ...
               'move', 'avg_des', 'mean_vel', 'std_dist2', 'std_contain', ...
               'min_dist', 'times'};

    % 读取两个方法的数据
    data_ms = loadAll(files_ms, metrics);
    data_km = loadAll(files_km, metrics);

    % 要绘图的指标及标签
    metric_list = {
        'covering_rates', 'Coverage Rate';
        'entering_rates', 'Entering Rate';
        'uniform_rates', 'Uniformity';
        'min_dist', 'Min Distance (m)';
    };

    % 配色列表：km（ours） 和 ms（mean-shift）
    color_main_km = [0.2, 0.4, 0.8];
    color_fill_km = [0.6, 0.7, 0.95];
    color_main_ms = [0.85, 0.33, 0.1];
    color_fill_ms = [1.0, 0.7, 0.5];

    % 保存路径
    saveDir = '/home/lhx/Desktop/trans7';
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    for k = 1:size(metric_list, 1)
        metric_name = metric_list{k, 1};
        y_label = metric_list{k, 2};

        % 获取插值后的数据矩阵
        [mean_km, min_km, max_km, t] = processMetric(data_km, metric_name);
        [mean_ms, min_ms, max_ms, ~] = processMetric(data_ms, metric_name);
        t = t * 0.05;

        % 绘图
        figure('Position', [100 100 750 600]);
        hold on;

        % ours 阴影 + 曲线
        fill([t, fliplr(t)], [max_km', fliplr(min_km')], ...
            color_fill_km, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        h1 = plot(t, mean_km, '-', 'Color', color_main_km, 'LineWidth', 3);

        % meanshift 阴影 + 曲线
        fill([t, fliplr(t)], [max_ms', fliplr(min_ms')], ...
            color_fill_ms, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        h2 = plot(t, mean_ms, '-', 'Color', color_main_ms, 'LineWidth', 3);

        % min_dist 虚线参考线
        if strcmp(metric_name, 'min_dist')
            yline(0.4, '--', 'Color', [0 0.6 0], 'LineWidth', 2);
        end

        % 美化
        grid on;
        box off;
        set(gca, 'LineWidth', 2.5);
        set(gca, 'FontSize', 24);
        xlabel('Time (s)', 'FontSize', 26);
        ylabel(y_label, 'FontSize', 26);
        xlim([0 1000 * 0.05]);
        legend([h1, h2], {'Ours', 'Mean-shift'}, 'Location', 'best', 'FontSize', 18);
        ax = gca;
        ax.FontName = 'Arial';
        ax.FontWeight = 'bold';

        % 保存图像
        savePath = fullfile(saveDir, [metric_name '.png']);
        print(gcf, savePath, '-dpng', '-r600');
        fprintf('图像已保存至: %s\n', savePath);
    end
end

% ====== 辅助函数：读取数据 ======
function data_struct = loadAll(file_list, metrics)
    numFiles = length(file_list);
    for m = 1:length(metrics)
        data_struct.(metrics{m}) = cell(numFiles, 1);
    end

    for i = 1:numFiles
        data = loadjson(file_list{i});
        for m = 1:length(metrics)
            data_struct.(metrics{m}){i} = data(:, m)';
        end
    end
end

% ====== 辅助函数：处理某一指标 ======
function [mean_vals, min_vals, max_vals, t] = processMetric(data_struct, metric_name)
    raw_data = data_struct.(metric_name);
    t_all = data_struct.times;
    t = t_all{1};  % 统一时间轴

    numFiles = length(raw_data);
    aligned = zeros(length(t), numFiles);
    for i = 1:numFiles
        aligned(:, i) = interp1(t_all{i}, raw_data{i}, t, 'linear', 'extrap');
    end

    mean_vals = mean(aligned, 2);
    min_vals = min(aligned, [], 2);
    max_vals = max(aligned, [], 2);
end
