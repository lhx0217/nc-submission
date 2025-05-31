function plotPos4_subplots_animated()
    % 文件和参数
    dataFiles = {'pos_km_dragon2.json', 'pos_ms_dragon2.json', 'heal_pos_dragon.json', 'sci_pos_dragon.json'};
    titles = {'Our Proposed', 'Mean-shift', 'Image Moment', 'Graph Similarity'};
    flipYs = [true, true, true, true];      % 均翻转Y
    swapXYs = [false, false, false, true]; % 仅pos_sci交换X,Y

    numPlots = 4;

    % 读取 graph 数据，图层固定
    graph = loadjson('dragon.json');
    graph(:, 2) = -graph(:, 2);

    % 预加载所有轨迹数据，判断最大步数
    posDatas = cell(numPlots, 1);
    maxSteps = 0;
    maxAgents = zeros(numPlots,1);

    for k = 1:numPlots
        posDatas{k} = loadjson(fullfile('./run_data', dataFiles{k}));
        maxSteps = max(maxSteps, size(posDatas{k}, 1));
        maxAgents(k) = size(posDatas{k}, 2);
    end

    % 可视化参数
    startColor = [0.35 0.35 0.35];
    endColor = [0 0.4470 0.7410];
    trajAlpha = 0.5;
    currentMarkerSize = 25;

    figure('Units','pixels','Position',[100 100 1600 400],'Color','w');

    % 初始化存储结构体数组，保存每个子图的handles和数据
    plots(numPlots) = struct();

    for k = 1:numPlots
        subplot(1,4,k);
        hold on; axis equal; view(2);
        grid on; box on;
        xlim([-35 35]); ylim([-35 35]);
        xlabel('X (m)', 'FontSize', 28)
        ylabel('Y (m)', 'FontSize', 28)
        ax = gca;
        ax.FontName = 'Arial';
        ax.FontWeight = 'bold';
        ax.LineWidth = 1.5;
        set(gca, 'FontSize', 18)
        title(titles{k}, 'Interpreter', 'none');

        % 画graph点
        scatter(graph(:,1), graph(:,2), 30, 'g', 'filled', 'MarkerFaceAlpha', 0.08);

        numAgents = maxAgents(k);
        traj_data = cell(numAgents,1);
        start_pts = zeros(numAgents, 2);
        end_pts = zeros(numAgents, 2);
        l = lines(numAgents);

        h_lines = gobjects(numAgents,1);
        h_start = scatter(nan(numAgents,1), nan(numAgents,1), 10, repmat(startColor,numAgents,1), 'filled');
        % h_end = scatter(nan(numAgents,1), nan(numAgents,1), 40, repmat(endColor,numAgents,1), 'filled');

        % 当前点handle (散点)
        h_robots = scatter(NaN, NaN, currentMarkerSize, [0 0.4470 0.7410], 'filled');

        % 预处理起点和终点
        for i = 1:numAgents
            pt_start = reshape(posDatas{k}(1,i,1:2), [1 2]);
            pt_end = reshape(posDatas{k}(end,i,1:2), [1 2]);
            if swapXYs(k)
                pt_start([1 2]) = pt_start([2 1]);
                pt_end([1 2]) = pt_end([2 1]);
            end
            if flipYs(k)
                pt_start(2) = -pt_start(2);
                pt_end(2) = -pt_end(2);
            end
            start_pts(i,:) = pt_start;
            end_pts(i,:) = pt_end;

            traj_data{i} = nan(maxSteps, 2);

            % 初始化轨迹线，颜色和透明度
            h_lines(i) = patch('XData', NaN, 'YData', NaN, ...
                   'EdgeColor', l(i,:), ...
                   'EdgeAlpha', 0.15, ...   % 透明度设置
                   'LineWidth', 1, ...
                   'FaceColor', 'none');
        end

        % 更新起点和终点散点数据
        set(h_start, 'XData', start_pts(:,1), 'YData', start_pts(:,2));
        % set(h_end, 'XData', end_pts(:,1), 'YData', end_pts(:,2));

        % 存储到结构体
        plots(k).h_lines = h_lines;
        plots(k).traj_data = traj_data;
        plots(k).posData = posDatas{k};
        plots(k).numAgents = numAgents;
        plots(k).flipY = flipYs(k);
        plots(k).swapXY = swapXYs(k);
        plots(k).h_robots = h_robots;
    end

    % GIF保存路径
    gifPath = '/home/lhx/Desktop/trans7/dragon_4plots_2D.gif';
    if exist(gifPath, 'file'), delete(gifPath); end

    % 动画主循环
    for t = 1:400
        disp(t);
        for k = 1:numPlots
            numAgents = plots(k).numAgents;
            posData = plots(k).posData;
            flipY = plots(k).flipY;
            swapXY = plots(k).swapXY;
            traj_data = plots(k).traj_data;
            h_lines = plots(k).h_lines;
            h_robots = plots(k).h_robots;

            pos_now = zeros(numAgents, 2);

            for i = 1:numAgents
                if t <= size(posData,1)
                    pt = reshape(posData(t,i,1:2), [1 2]);
                    if swapXY
                        pt([1 2]) = pt([2 1]);
                    end
                    if flipY
                        pt(2) = -pt(2);
                    end
                else
                    pt = [NaN NaN];
                end

                traj_data{i}(t, :) = pt;
                data = traj_data{i}(1:t, :);
                data = [data; [NaN, NaN]];
                set(h_lines(i), 'XData', data(:,1), 'YData', data(:,2));
                pos_now(i, :) = pt;
            end

            set(h_robots, 'XData', pos_now(:,1), 'YData', pos_now(:,2));

            plots(k).traj_data = traj_data;
        end

        drawnow;

        % 采集全图帧写入GIF
        frame = getframe(gcf);
        im = frame2im(frame);
        [A,map] = rgb2ind(im,256);
        if t == 1
            imwrite(A,map,gifPath,'gif','LoopCount',Inf,'DelayTime',0.05);
        else
            imwrite(A,map,gifPath,'gif','WriteMode','append','DelayTime',0.05);
        end
    end

    fprintf('4子图组合 2D GIF 已保存至: %s\n', gifPath);
end
