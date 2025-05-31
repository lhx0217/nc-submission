function plotRates()
    % 文件名
    files = {'./run_data/rate_km_sense5.2.json'};
    
    % 初始化数据存储
    numFiles = length(files);
    entering_rates = cell(numFiles, 1);
    covering_rates = cell(numFiles, 1);
    uniform_rates = cell(numFiles, 1);
    move = cell(numFiles, 1);
    avg_des = cell(numFiles, 1);
    mean_vel = cell(numFiles, 1);
    std_dist2 = cell(numFiles, 1);
    std_contain = cell(numFiles, 1);
    times = cell(numFiles, 1);
    min_dist = cell(numFiles, 1);
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
        times{i} = data(:, 10)';
    end
    
    % legends = {'km+ms', 'ms', 'nature(ms+协商方案)'};
    legends = {'km', 'ms'};
    ylabels = {'覆盖率', '进入率', '平均距离差(米)', '移动距离', '协商距离', '平均速度', '邻居距离方差', '所属面积方差', '最小距离'};
    ylabels2 = {'进入指令', 'Kmeans指令', '面积指令', '距离指令', '避障指令', '总指令'};
    
    % 创建一个窗口
    figure;
    
    % 绘制覆盖率图
    subplot(3, 3, 1); % 3行1列的第1个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, covering_rates{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{1});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);
    
    % 绘制进入率图
    subplot(3, 3, 2); % 3行1列的第2个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, entering_rates{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{2});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);
    
    % 绘制平均距离差图
    subplot(3, 3, 3); % 3行1列的第3个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, uniform_rates{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{3});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);

    subplot(3, 3, 4); % 3行2列的第1个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, move{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{4});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);

    subplot(3, 3, 5); % 3行2列的第1个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, avg_des{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{5});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);

    subplot(3, 3, 6); % 3行2列的第1个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, mean_vel{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{6});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);

    subplot(3, 3, 7); % 3行2列的第1个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, std_dist2{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{7});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);

    subplot(3, 3, 8); % 3行2列的第1个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, std_contain{i}, 'LineWidth', 1);
    end
    legend(legends);
    xlabel('时间(0.05s)');
    ylabel(ylabels{8});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);

    subplot(3, 3, 9); % 3行2列的第9个位置
    hold on;
    for i = 1:numFiles
        plot(times{i}, min_dist{i}, 'LineWidth', 1);
    end
    xlabel('时间(0.05s)');
    ylabel(ylabels{9});
    grid on;
    box off;
    set(gca, 'LineWidth', 1);
    % 绘制深绿色的y=1的横线
    yline = 1; % 横线的位置
    maxTimeLength = max(cellfun(@length, times));
    plot(linspace(min(times{1}), max(times{end}), maxTimeLength), yline * ones(maxTimeLength, 1), 'Color', '#0DB719', 'LineWidth', 1.5);

    % 标注所有在y=1线下方的点
    for i = 1:numFiles
        for j = 1:length(min_dist{i})
            if min_dist{i}(j) < yline
                % 使用scatter标注点
                scatter(times{i}(j), min_dist{i}(j), 10, 'filled', 'MarkerFaceColor', 'red', 'LineWidth', 0.5);
            end
        end
    end

    % 创建一个窗口, 绘制指令占比
    figure;
    files = {'./run_data/comm_km_sense5.2.json'};
    % 初始化数据存储
    numFiles = length(files);
    comm_percent = cell(numFiles, 1);
    for i = 1:numFiles
        comm_percent{i} = loadjson(files{i});
    end
    for p = 1:6
        subplot(2, 3, p);
        for i = 1:numFiles
            plot(times{i}, comm_percent{i}(:,p), 'LineWidth', 1);
            hold on;
        end
        legend(legends);
        xlabel('时间(0.05s)');
        ylabel(ylabels2{p});
        grid on;
        box off;
    end

end