%% 参数设置
robot_counts = [50, 100, 200, 500];       % 横坐标的机器人数量
methods = {'time_km', 'time_ms', 'heal_time', 'sci_time'}; % 方法名称前缀
num_robots = length(robot_counts);         % 机器人数量类型数
num_methods = length(methods);             % 方法种类数
n = 900;                                  % 每个实验的循环次数
confidence_level = 0.95;                   % 置信水平

%% 数据加载与计算
mean_time = zeros(num_robots, num_methods); % 均值矩阵初始化
ci = zeros(num_robots, num_methods);        % 置信区间半宽矩阵

for robot_idx = 1:num_robots
    for method_idx = 1:num_methods
        % 构造文件名（根据实际路径修改）
        filename = sprintf('./run_data/%s_%d.json', methods{method_idx}, robot_counts(robot_idx));
        disp(filename);
        % 加载数据文件（假设数据为1×1000行向量）
        data = loadjson(filename);
        % data = data(51:950);
        data = diff(data) / robot_counts(robot_idx);
        % 计算均值和标准差
        mean_val = mean(data);
        std_val = std(data);
        
        % 计算标准误和置信区间（使用t分布）
        se = std_val / sqrt(n);
        t_critical = 2.262; % t临界值
        ci_half_width = t_critical * se;
        
        % 存储结果
        mean_time(robot_idx, method_idx) = mean_val;
        ci(robot_idx, method_idx) = ci_half_width;
    end
end

%% 绘制带置信区间的柱状图
figure;
bar_handle = bar(mean_time);  % 绘制基础柱状图
hold on;

% 设置图形样式
colors = lines(num_methods); % 为不同方法分配颜色
for m = 1:num_methods
    bar_handle(m).FaceColor = colors(m,:);
    bar_handle(m).FaceAlpha = 0.8;
end

% 添加误差条（置信区间）
for robot_idx = 1:num_robots
    for method_idx = 1:num_methods
        % 获取柱子的x坐标
        x_pos = bar_handle(method_idx).XEndPoints(robot_idx);
        
        % 绘制误差条
        errorbar(x_pos, mean_time(robot_idx, method_idx),...
                ci(robot_idx, method_idx),...
                'k', 'LineWidth', 1.2, 'CapSize', 15);
    end
end

% 添加标签和标题
set(gca, 'XTick', 1:num_robots, 'XTickLabel', robot_counts);
xlabel('机器人数量','FontSize',12);
ylabel('运行时间（秒）','FontSize',12);
title('不同方法运行时间对比（95%置信区间）','FontSize',14);
legend(methods, 'Location', 'northwest');
grid on;

% 优化图形显示
set(gca, 'FontSize',11, 'LineWidth',1.2);
set(gcf, 'Position', [100, 100, 800, 500]);