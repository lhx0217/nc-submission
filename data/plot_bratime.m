%% 参数设置
robot_counts = [50, 100, 200, 500];       % 横坐标的机器人数量
methods = {'time_km', 'time_ms', 'heal_time', 'sci_time'}; % 方法名称前缀
num_robots = length(robot_counts);        % 机器人数量类型数
num_methods = length(methods);             % 方法种类数
n = 900;                                  % 每个实验的循环次数
confidence_level = 0.95;                  % 置信水平

%% 数据加载与计算
mean_time = zeros(num_robots, num_methods); % 均值矩阵初始化
ci = zeros(num_robots, num_methods);       % 置信区间半宽矩阵

for robot_idx = 1:num_robots
    for method_idx = 1:num_methods
        % 构造文件名
        filename = sprintf('./run_data/%s_%d.json', methods{method_idx}, robot_counts(robot_idx));
        
        % 加载数据文件
        data = loadjson(filename);
        data = diff(data) / robot_counts(robot_idx); % 时间差归一化
        % 计算统计量
        mean_val = mean(data);
        std_val = std(data);
        
        % 计算标准误和置信区间
        se = std_val / sqrt(n);
        t_critical = 2.262; % 对应n=10的t值（建议改用自动计算：tinv(0.975, n-1))
        ci_half_width = t_critical * se;
        
        % 存储结果
        mean_time(robot_idx, method_idx) = mean_val;
        ci(robot_idx, method_idx) = ci_half_width;
    end
end

%% 绘制带置信区间的柱状图
figure('Position', [100 100 800 600])
bar_handle = bar(mean_time);  % 绘制基础柱状图
hold on;

% 设置对数坐标轴
set(gca, 'YScale', 'log');
ylim([1e-4, 0.5]); % 设置纵坐标范围
yticks(10.^[-4:0]);  % 显示10^-3到10^0的刻度
yticklabels({'10^{-4}','10^{-3}','10^{-2}','10^{-1}'});
set(gca, 'YMinorTick','off'); % 关闭次要刻度
method_names = {'Our Proposed', 'Mean-shift', 'Image Moment', 'Graph Similarity'}; % 自定义图例名称

% 设置图形样式
colors = lines(num_methods); % 为不同方法分配颜色
for m = 1:num_methods
    bar_handle(m).FaceColor = colors(m,:);
    bar_handle(m).FaceAlpha = 0.9;
    bar_handle(m).DisplayName = method_names{m}; % 为每个柱状图系列指定名称
end

% 添加误差条（置信区间）
for robot_idx = 1:num_robots
    for method_idx = 1:num_methods
        % 获取柱子的x坐标
        x_pos = bar_handle(method_idx).XEndPoints(robot_idx);
        
        % 绘制误差条
        errorbar(x_pos, mean_time(robot_idx, method_idx),...
                ci(robot_idx, method_idx),...
                'k', 'LineWidth', 1, 'CapSize', 15);
    end
end
    
% 添加标签和标题
set(gca, 'XTick', 1:num_robots, 'XTickLabel', robot_counts);
xlabel('Swarm Size','FontSize',28);
ylabel('Time (s)','FontSize',28);
% legend(methods, 'Location', 'northwest');
grid on;
set(gca, 'XGrid', 'off'); % 关闭垂直主网格线
set(gca, 'XMinorGrid', 'off'); % 关闭垂直次要网格线
             % 坐标轴线宽同步加粗
legend(bar_handle, method_names,'Location', 'northeast',...
       'FontSize', 20,...          % 适当减小字体
       'EdgeColor',[0.2 0.2 0.2],... % 灰色边框
       'Box','on',...              % 显示边框
       'Orientation','vertical');  % 垂直排列
ax = gca;
ax.FontName = 'Arial';
ax.FontWeight = 'bold';
ax.LineWidth = 3;
set(gca, 'FontSize', 28)