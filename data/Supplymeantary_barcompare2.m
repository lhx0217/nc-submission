function plotAvoidanceComparison()
    methods = {'km', 'ms'};
    avoid_dists = [0, 0.4, 0.8, 1.2, 1.6, 2.0];
    repeats = 1:3;

    % 指标列索引
    metric_cols = struct( ...
        'covering_rates', 2, ...
        'entering_rates', 1, ...
        'uniform_rates', 3, ...
        'min_dist', 9 ...
    );

    metric_labels = struct( ...
        'covering_rates', 'Coverage Rate', ...
        'entering_rates', 'Entering Rate', ...
        'uniform_rates', 'Uniformity', ...
        'min_dist', 'Min Distance (m)' ...
    );

    % 配色
    color_pairs = {
        [0.2 0.4 0.9], [0.5 0.7 1.0];   % 蓝
        [0.85 0.2 0.2], [1.0 0.6 0.6];  % 红
        [0.2 0.6 0.3], [0.6 0.9 0.6];   % 绿
        [0.5 0.3 0.7], [0.8 0.6 0.9];   % 紫
    };

    metric_names = fieldnames(metric_cols);

    for k = 1:length(metric_names)
        metric = metric_names{k};
        col_idx = metric_cols.(metric);
        ylabel_str = metric_labels.(metric);
        color_ours = color_pairs{k,1};
        color_ms   = color_pairs{k,2};

         % 读取数据
        values = nan(2, length(avoid_dists), length(repeats));
        for m = 1:length(methods)
            method = methods{m};
            for d = 1:length(avoid_dists)
                dist = avoid_dists(d);
                dist_str = strrep(num2str(dist), '.', '.');
                for r = repeats
                    file = sprintf('./run_data/rate_%s_avoid%s.%d.json', method, dist_str, r);
                    if isfile(file)
                        try
                            raw = loadjson(file);
                            vec = raw(:, col_idx);
                            if strcmp(metric, 'min_dist')
                                % 取整个vec中的最小值
                                values(m, d, r) = min(vec);
                            else
                                % 默认取最后一个值
                                values(m, d, r) = vec(end);
                            end
                        catch
                            warning('Error reading %s', file);
                        end
                    else
                        warning('Missing file: %s', file);
                    end
                end
            end
        end


        % 统计均值与误差
        mean_vals = squeeze(mean(values, 3, 'omitnan'));
        min_vals  = squeeze(min(values, [], 3));
        max_vals  = squeeze(max(values, [], 3));
        lowers = mean_vals - min_vals;
        uppers = max_vals - mean_vals;

        % 绘图
        figure('Position', [100, 100, 1200, 600]); hold on;
        b = bar(mean_vals', 'grouped');  % 不传 x，自动使用1:N

        % 设置颜色和透明度
        b(1).FaceColor = color_ours;
        b(2).FaceColor = color_ms;
        b(1).FaceAlpha = 0.7;
        b(2).FaceAlpha = 0.7;

        % 添加误差棒
        ngroups = size(mean_vals, 2);
        nbars = size(mean_vals, 1);
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1)*groupwidth/(2*nbars);
            errorbar(x, mean_vals(i,:), lowers(i,:), uppers(i,:), ...
                'k', 'linestyle', 'none', 'LineWidth', 1.8, 'CapSize', 6);
        end

        % 美化
        grid on;
        box off;
        set(gca, 'FontSize', 22, 'LineWidth', 2.5);
        xlabel('Avoidance Distance (m)', 'FontSize', 26);
        ylabel(ylabel_str, 'FontSize', 26);
        title(ylabel_str, 'FontSize', 28);
        xticks(1:length(avoid_dists));
        xticklabels(string(avoid_dists));
        legend({'Ours', 'Mean-Shift'}, 'FontSize', 20, 'Location', 'best');
        ax = gca;
        ax.FontName = 'Arial';
        ax.FontWeight = 'bold';

        % 保存图像
        saveDir = './avoidance_plots';
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        savePath = sprintf('%s/%s.png', saveDir, metric);
        print(gcf, savePath, '-dpng', '-r600');
        fprintf('图像已保存：%s\n', savePath);
    end
end
