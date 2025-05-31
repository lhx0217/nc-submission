%% 参数设置
methods = {'rate_km', 'rate_ms', 'heal_cover', 'sci_cover'}; % 方法前缀
num_runs = 5;      % 每个方法的运行次数
data_length = 1000; % 数据长度

%% 颜色和线型定义
colors = [0 0.45 0.74;   % 蓝
          0.47 0.67 0.19;% 绿
          0.85 0.33 0.10;% 红
          0.49 0.18 0.56];% 紫
line_styles = {'-', '--', ':', '-.'};

%% 准备图形
figure('Position', [100 100 800 600])
hold on
x = 1:data_length;
xlim([0 100]);
ylim([0 1]);

%% 主循环处理每个方法
plot_handles = gobjects(1,4); % 预分配图形对象数组
for method_idx = 1:4
    % 读取所有运行数据
    all_data = zeros(num_runs, data_length);
    for run_num = 1:num_runs
        filename = sprintf('./run_data/%s_%d.json', methods{method_idx}, run_num);
        json_data = loadjson(filename);
        if method_idx < 3
            json_data = json_data(:, 2)';
        end
        all_data(run_num, :) = json_data;
    end
    
    % 计算统计量
    mean_val = mean(all_data, 1);
    std_val = std(all_data, 0, 1);
    n = size(all_data, 1);
    se = std_val / sqrt(n);
    t_critical = 2.262;
    ci = t_critical * se;

    % 绘制置信区间（不加入图例）
    fill_color = colors(method_idx, :) + (1 - colors(method_idx, :)) * 0.3;
    fill_x = [x, fliplr(x)];
    fill_y = [mean_val - ci, fliplr(mean_val + ci)];
    fill(fill_x * 0.1, fill_y, fill_color, 'EdgeColor', 'none', 'FaceAlpha', 0.4)

    % 绘制均值曲线并设置图例名称
    h = plot(x * 0.1, mean_val, 'Color', colors(method_idx, :),...
           'LineWidth', 3, 'LineStyle', line_styles{method_idx});
    
    % 为每条曲线指定图例名称
    switch method_idx
        case 1
            name = 'Our Proposed';
        case 2
            name = 'Mean-shift';
        case 3
            name = 'Image Moment';
        case 4
            name = 'Graph Similarity';
    end
    set(h, 'DisplayName', name);
    plot_handles(method_idx) = h;
end

%% 图形美化
xlabel('Time(s)', 'FontSize', 28)
ylabel('Coverage Rate', 'FontSize', 28)
ax = gca;
ax.FontName = 'Arial';
ax.FontWeight = 'bold';
ax.LineWidth = 3;

% 设置图例（仅显示四条曲线）
legend_labels = {'Our Proposed','Mean-shift','Image Moment','Graph Similarity'};
legend(plot_handles, legend_labels,...
      'Location', 'best',...
      'FontSize', 20);

grid on
box on
set(gca, 'FontSize', 28)
hold off