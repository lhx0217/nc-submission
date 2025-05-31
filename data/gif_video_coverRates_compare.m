function plotMultiCurveGif()
    %% 参数设置
    methods = {'rate_km', 'rate_ms', 'heal_cover', 'sci_cover'};
    method_names = {'Our Proposed', 'Mean-shift', 'Image Moment', 'Graph Similarity'};
    num_methods = length(methods);
    data_length = 400;

    %% 配色和线型
    colors = [0 0.45 0.74;   % 蓝
              0.47 0.67 0.19;% 绿
              0.85 0.33 0.10;% 红
              0.49 0.18 0.56];% 紫
    line_styles = {'-', '--', ':', '-.'};

    %% 数据读取（每种方法只读一组）
    all_mean = zeros(num_methods, data_length);
    for method_idx = 1:num_methods
        if method_idx < 3
            filename = sprintf('./run_data/%s_dragon2.json', methods{method_idx});
            json_data = loadjson(filename);
        else
            filename = sprintf('./run_data/%s_dragon.json', methods{method_idx});
            json_data = loadjson(filename);
        end
        if method_idx < 3
            json_data = json_data(:, 2)'; % 取第2列
        end
        all_mean(method_idx, :) = json_data(1:400);
    end

    %% 准备GIF保存路径
    gifPath = './run_data/method_compare.gif';
    if exist(gifPath, 'file'), delete(gifPath); end

    %% 创建图像
    figure('Position', [100, 100, 800, 600]);

    x = (1:data_length) * 0.1;
    for frame_idx = 1:data_length
        clf;
        hold on;
        xlim([0, 40]);
        ylim([0, 1]);
        for method_idx = 1:num_methods
            plot(x(1:frame_idx), all_mean(method_idx, 1:frame_idx), ...
                 'Color', colors(method_idx, :), ...
                 'LineWidth', 2.5, ...
                 'LineStyle', line_styles{method_idx}, ...
                 'DisplayName', method_names{method_idx});
        end

        % 美化
        xlabel('Time(s)', 'FontSize', 24);
        ylabel('Coverage Rate', 'FontSize', 24);
        grid on;
        box on;
        ax = gca;
        ax.FontName = 'Arial';
        ax.FontWeight = 'bold';
        ax.LineWidth = 2.5;
        ax.FontSize = 22;
        legend('Location', 'southeast', 'FontSize', 16);

        drawnow;
        frame = getframe(gcf);
        im = frame2im(frame);
        [A, map] = rgb2ind(im, 256);
        if frame_idx == 1
            imwrite(A, map, gifPath, 'gif', 'LoopCount', Inf, 'DelayTime', 0.05);
        else
            imwrite(A, map, gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
        end
    end

    close;
    fprintf('GIF saved to %s\n', gifPath);
end
