clearvars;

parallel.gpu.enableCUDAForwardCompatibility(true);
try gpuDevice(1); catch, warning('GPU Reset Failed'); end

fprintf('----------------------------------------------------------------\n');
fprintf('Shear Wave Simulation On Skin');
fprintf('----------------------------------------------------------------\n');

%% --- Simulation Parameters ---
param.cfl = 0.3;
param.source_threshold = 1e-6; 

medium_air.sound_speed = 343;     % [m/s]
medium_air.density     = 1.21;    % [kg/m^3]

medium_skin.sound_speed_compression = 1540; % [m/s] Soft tissue (Physical)
medium_skin.sound_speed_compression_sim = 100; % [m/s] Optimized for simulation (sufficient for shear separation)
medium_skin.sound_speed_shear       = 5.0;  % [m/s]
medium_skin.density                 = 923.5; % [kg/m^3]
medium_skin.alpha_coeff_shear       = 10;
medium_skin.alpha_power_shear       = 2.0;
medium_skin.alpha_coeff_compression = 0.1;  % [dB/(MHz^y cm)]
medium_skin.alpha_power_compression = 1.5;

medium_skin.sponge_layers = 20;   % Absorption layer thickness (grid points)
medium_skin.sponge_alpha_max = 30; % Max absorption [dB/(MHz^y cm)]

target.update_Hz = 200;     % [Hz] Repetition Frequency (Unified for all methods)
target.num_focal_points = 6; % [Integer] Number of discrete focal points on the ring

target.cycle_period = 1 / target.update_Hz; % [s]
target.sim_duration = target.cycle_period * 1; % [s] 1 cycle
target.mach_number = 1.2;
rayleigh_factor = 0.95;
target.v_focus = target.mach_number * medium_skin.sound_speed_shear * rayleigh_factor;

grid_cfg.dx = 1.0e-3; % [m]
grid_cfg.ramp_length_pct = 0.05; % Percentage of signal to ramp up/down (5%)

grid_cfg.Lx = 100e-3; % [m]
grid_cfg.Ly = 100e-3; % [m]
grid_cfg.Lz_skin = 40e-3;                 % Skin domain depth

dt = param.cfl * grid_cfg.dx / medium_skin.sound_speed_compression_sim; 

%% 2. Grid Setup

% Space
Nx = round(grid_cfg.Lx / grid_cfg.dx);
Ny = round(grid_cfg.Ly / grid_cfg.dx);

Nz = round(grid_cfg.Lz_skin / grid_cfg.dx);

% Ensure even numbers for FFT efficiency
if mod(Nx,2), Nx=Nx+1; end
if mod(Ny,2), Ny=Ny+1; end
if mod(Nz,2), Nz=Nz+1; end

kgrid = kWaveGrid(Nx, grid_cfg.dx, Ny, grid_cfg.dx, Nz, grid_cfg.dx);

fprintf('Skin Grid: %d x %d x %d\n', Nx, Ny, Nz);

% Time
Nt = ceil(target.sim_duration/dt);

t_vec = (0: Nt-1)*dt;

kgrid.setTime(length(t_vec), dt);

fprintf('Total Time Steps: %d, Single Step: %f s\n', Nt, dt);

% medium properties

medium = struct(); % Initialize clean struct

medium.sound_speed_compression = medium_skin.sound_speed_compression_sim; 
medium.sound_speed_shear       = medium_skin.sound_speed_shear;
medium.density                 = medium_skin.density;

medium.alpha_coeff_compression = medium_skin.alpha_coeff_compression* ones(Nx, Ny, Nz, 'single');

medium.alpha_coeff_shear = medium_skin.alpha_coeff_shear* ones(Nx, Ny, Nz, 'single');

% Sponge Layer (Bottom Absorption)
sponge_layers = medium_skin.sponge_layers;
sponge_profile = reshape(medium_skin.sponge_alpha_max * linspace(0, 1, sponge_layers).^2, [1, 1, sponge_layers]);
medium.alpha_coeff_compression(:, :, end-sponge_layers+1:end) = medium.alpha_coeff_compression(:, :, end-sponge_layers+1:end) + sponge_profile;
medium.alpha_coeff_shear(:, :, end-sponge_layers+1:end) = medium.alpha_coeff_shear(:, :, end-sponge_layers+1:end) + sponge_profile;


%% 3. Signal Generate
source = struct();
ring_start_radius = target.v_focus / target.update_Hz;

[X_grid, Y_grid] = ndgrid((0:Nx-1)*grid_cfg.dx, (0:Ny-1)*grid_cfg.dx);
X_grid = X_grid - mean(X_grid(:));
Y_grid = Y_grid - mean(Y_grid(:));
R_map = sqrt(X_grid.^2 + Y_grid.^2);
Theta_map = atan2(Y_grid, X_grid);

source_field_2d = zeros(Nx, Ny, Nt, 'single');
ring_width = 4.25e-3;

min_R_clamp = 1e-3; 
base_amplitude = 500; 

for i = 1:Nt
    t = t_vec(i);
    phase = mod(t, target.cycle_period) / target.cycle_period;

    % STM focusing, make a focus point traveling along a circular path
    % focus_phase = 2 * pi * phase; % Phase of the focus point
    % focus_R = target.v_focus / (2 * pi * target.update_Hz);
    % focus_x = focus_R * cos(focus_phase);
    % focus_y = focus_R * sin(focus_phase);
    % dist_to_focus = sqrt((X_grid - focus_x).^2 + (Y_grid - focus_y).^2);
    % gaussian_point = exp(-(dist_to_focus.^2) / (2 * ring_width^2));
    % source_field_2d(:, :, i) = gaussian_point;


    % SWI
    % current_R = ring_start_radius * (1 - phase);
    % current_ramp = -(2*phase - 1)^20 + 1;
    % if current_R < grid_cfg.dx * 0.5
    %     amplitude_scaling = 0; % 到达中心后关断源
    % else
    %     scaling_factor = sqrt(ring_start_radius / max(current_R, min_R_clamp));
    %     amplitude_scaling = base_amplitude * scaling_factor;
    % end
    % gaussian_ring = exp(-(R_map - current_R).^2 / (2 * ring_width^2));
    % source_field_2d(:, :, i) = gaussian_ring * amplitude_scaling * current_ramp;

    % Rotational Shear Ultrasound Tactile RUST (Discrete Sampling)
    spiral_phase = 2 * pi * phase; % Current angle of the pointer (0 to 2pi)
    
    accumulated_field = zeros(Nx, Ny, 'single');
    
    for k = 1:target.num_focal_points
        % 1. Determine the fixed angle for this sampling point
        % Points are uniformly distributed on the ring: 0, 2pi/N, 4pi/N...
        point_angle = (k - 1) * (2 * pi / target.num_focal_points);
        
        % 2. Calculate the "lag" for this specific point
        % When the pointer (spiral_phase) passes point_angle, lag is 0 -> Radius is max
        % As pointer moves away, lag increases -> Radius decreases
        lag_angle = mod(spiral_phase - point_angle, 2*pi);
        
        % 3. Calculate current radius for this point
        % r = R_start * (1 - lag / 2pi)
        r_point = ring_start_radius * (1 - lag_angle / (2*pi));
        
        % 4. Determine (x, y) coordinates of this focal point
        p_x = r_point * cos(point_angle);
        p_y = r_point * sin(point_angle);
        
        % 5. Generate Gaussian spot at this position
        dist_sq = (X_grid - p_x).^2 + (Y_grid - p_y).^2;
        gaussian_spot = exp(-dist_sq / (2 * ring_width^2));
        
        % Accumulate the field
        accumulated_field = accumulated_field + gaussian_spot;
    end
    
    % Apply amplitude
    source_field_2d(:, :, i) = accumulated_field * base_amplitude;
    
end

s_mask_sum = max(source_field_2d, [], 3);

source.s_mask = zeros(Nx, Ny, Nz); 
source.s_mask(:, :, 1) = (s_mask_sum > 1e-3);
source.s_mask = (source.s_mask == 1);
active_indices_2d = find(source.s_mask(:, :, 1));
num_active_points = length(active_indices_2d);

F_full = reshape(source_field_2d, Nx*Ny, Nt);
source_amp_stress = 1e2;  % [Pa] 试探值，可根据结果调整
source.szz = -F_full(active_indices_2d, :) ; 

% 清除旧的变量以防万一
if isfield(source, 'u_mask'), source = rmfield(source, 'u_mask'); end
if isfield(source, 'uz'), source = rmfield(source, 'uz'); end

clear source_field_2d F_full s_mask_sum;

%% Sensor Setup

sensor = struct();

mask_cube = false(Nx, Ny, Nz);
mask_surf(:, :, 2) = true;
depth_under_skin = 5e-3; % [mm] Sensor depth below surface
record_depth_idx = ceil(depth_under_skin / grid_cfg.dx);
record_depth_idx = min(record_depth_idx, Nz);
mask_cube(:, :, 1:record_depth_idx) = true;
sensor.mask = mask_cube;
sensor.record = {'u'};

%% 6. Run Simulation
if gpuDeviceCount > 0
    data_cast = 'gpuArray-single';
else
    data_cast = 'single';
end

% Sponge Layer
input_args = {'PMLSize', [20, 20, 10], 'PMLInside', false, ...
              'PMLAlpha', [2, 2, 0], ... 
              'DataCast', data_cast, ...
              'PlotPML', false, 'PlotLayout', false};

sensor_data = pstdElastic3D(kgrid, medium, source, sensor, input_args{:});

sensor_data.ux = gather(sensor_data.ux);
sensor_data.uy = gather(sensor_data.uy);
sensor_data.uz = gather(sensor_data.uz);

%% 7. Post-Process

num_sensor_points = sum(sensor.mask(:));
if num_sensor_points == Nx * Ny * record_depth_idx
    % 重排数据: Linear -> [Nx, Ny, Nz_rec, Nt]
    % 注意：k-Wave/Matlab 是列优先，先填 x，再填 y
    sensor_data.ux  = reshape(sensor_data.ux,  [Nx, Ny, record_depth_idx, Nt]);
    sensor_data.uy  = reshape(sensor_data.uy,  [Nx, Ny, record_depth_idx, Nt]);
    sensor_data.uz  = reshape(sensor_data.uz,  [Nx, Ny, record_depth_idx, Nt]);
else
    error('Sensor mask size does not match expected reshape dimensions.');
end

% 1. 积分速度得到位移 (Displacement) 
% 使用 cumtrapz 提高精度
u_x = cumtrapz(t_vec, sensor_data.ux, 4); 
u_y = cumtrapz(t_vec, sensor_data.uy, 4);
u_z = cumtrapz(t_vec, sensor_data.uz, 4);

u_z_max = max(abs(u_z), [], 4); 

% 2. 计算空间梯度 (Finite Difference) 
% 计算剪切应变 epsilon_xz 和 epsilon_yz

% 预分配内存 
[Nx_s, Ny_s, Nz_s, Nt_s] = size(u_x); 
epsilon_xz = zeros(Nx_s, Ny_s, Nz_s, Nt_s, 'single'); 
epsilon_yz = zeros(Nx_s, Ny_s, Nz_s, Nt_s, 'single'); 

fprintf('Calculating Shear Strain from Displacement Gradients...\n');

for t = 1:Nt_s 
    % 提取当前时刻的 3D 体数据 
    Ux_vol = u_x(:, :, :, t); 
    Uy_vol = u_y(:, :, :, t); 
    Uz_vol = u_z(:, :, :, t); 
    
    % 计算导数 
    [dUx_dy, dUx_dx, dUx_dz] = gradient(Ux_vol, grid_cfg.dx); 
    [dUy_dy, dUy_dx, dUy_dz] = gradient(Uy_vol, grid_cfg.dx); 
    [dUz_dy, dUz_dx, dUz_dz] = gradient(Uz_vol, grid_cfg.dx); 
    
    % 计算剪切应变张量分量 
    % epsilon_xz = 0.5 * (du_x/dz + du_z/dx)
    epsilon_xz(:, :, :, t) = 0.5 * (dUx_dz + dUz_dx); 
    
    % epsilon_yz = 0.5 * (du_y/dz + du_z/dy)
    epsilon_yz(:, :, :, t) = 0.5 * (dUy_dz + dUz_dy); 
end 

epsilon_shear_mod = sqrt(epsilon_xz.^2 + epsilon_yz.^2);

%% 8. Advanced Visualization (RUST Analysis)
fprintf('----------------------------------------------------------------\n');
fprintf('Generating Advanced Visualizations...\n');
fprintf('----------------------------------------------------------------\n');

% Prepare Data for Surface (z=1)
Exz_surf = squeeze(epsilon_xz(:, :, 1, :)); % [Nx, Ny, Nt]
Eyz_surf = squeeze(epsilon_yz(:, :, 1, :)); % [Nx, Ny, Nt]
E_mod_surf = squeeze(epsilon_shear_mod(:, :, 1, :)); % [Nx, Ny, Nt]

%% 8.1 Hodograph: Center Point Vector Trajectory (Optimized)
% 核心验证：中心点矢量轨迹图
fprintf('[1/5] Generating Hodograph (Nature Style)...\n');

center_idx_x = round(Nx/2);
center_idx_y = round(Ny/2);

% Extract center traces (using mean of 3x3 to reduce noise)
range_x = center_idx_x-1 : center_idx_x+1;
range_y = center_idx_y-1 : center_idx_y+1;

trace_xz = squeeze(mean(mean(Exz_surf(range_x, range_y, :), 1), 2));
trace_yz = squeeze(mean(mean(Eyz_surf(range_x, range_y, :), 1), 2));

fig_hodo = figure('Name', 'Hodograph', 'Color', 'w', 'Visible', 'off','Position', [100, 100, 500, 500]);

% % 1. Plot Ideal Circle (Reference)
max_amp = max(sqrt(trace_xz.^2 + trace_yz.^2));
% theta_circ = linspace(0, 2*pi, 100);
% plot(max_amp * cos(theta_circ), max_amp * sin(theta_circ), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5); 
hold on;

% 2. Plot Trajectory (Solid Line, Dark Color)
plot(trace_xz, trace_yz, 'Color', [0 0.2 0.6], 'LineWidth', 2); % Dark Blue

% 3. Add Time Arrows (Quarter Intervals)
arrow_indices = round(linspace(1, length(t_vec), 10)); 
arrow_indices = arrow_indices(1:end-1); % 0, T/4, T/2, 3T/4
for idx = arrow_indices
    % Normalized direction
    dx = trace_xz(idx+1) - trace_xz(idx);
    dy = trace_yz(idx+1) - trace_yz(idx);
    len = sqrt(dx^2 + dy^2);
    if len > 0
        quiver(trace_xz(idx), trace_yz(idx), dx/len*max_amp*0.15, dy/len*max_amp*0.15, ...
            'Color', 'k', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    end
end

% 4. Start/End Labels
plot(trace_xz(1), trace_yz(1), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
text(trace_xz(1)+max_amp*0.05, trace_yz(1), 'Start', 'FontName', 'Arial', 'FontSize', 8);

plot(trace_xz(end), trace_yz(end), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
text(trace_xz(end)+max_amp*0.05, trace_yz(end), 'End', 'FontName', 'Arial', 'FontSize', 8);

hold off;
axis equal; 
box on; % Keep box for axes
grid on;

% Nature Style Formatting
set(gca, 'FontName', 'Arial', 'FontSize', 9);
xlabel('Shear Strain \epsilon_{xz}', 'FontSize', 9);
ylabel('Shear Strain \epsilon_{yz}', 'FontSize', 9);
title(''); % Remove title for paper

% Export as EPS
exportgraphics(fig_hodo, '1_Hodograph_Vector_Trajectory.svg', 'ContentType', 'vector');
close(fig_hodo);


%% 8.2 Dynamic Vector Field Animation (Quiver)
% % 动态矢量场图
% fprintf('[2/5] Generating Quiver Animation...\n');

% % Downsample for quiver plot
% skip = 4;
% % Define ROI indices for plotting
% x_roi_idx = find(abs(kgrid.x_vec) < 10e-3); % Center 10mm span
% y_roi_idx = find(abs(kgrid.y_vec) < 10e-3);
% x_roi_idx = x_roi_idx(1:skip:end);
% y_roi_idx = y_roi_idx(1:skip:end);

% [Y_q, X_q] = meshgrid(kgrid.y_vec(y_roi_idx)*1000, kgrid.x_vec(x_roi_idx)*1000);

% v_quiver = VideoWriter('2_Vector_Field_Dynamics.avi', 'Motion JPEG AVI');
% v_quiver.FrameRate = 60;
% open(v_quiver);

% fig_quiv = figure('Name', 'Vector Field', 'Color', 'w', 'Visible', 'off');
% set(fig_quiv, 'Position', [100, 100, 600, 500]);

% % Fixed Axes Setup
% ax_q = axes('Parent', fig_quiv);
% axis(ax_q, 'equal');
% grid(ax_q, 'on');
% box(ax_q, 'on');

% % Calculate limits with a small margin to ensure arrows fit
% margin = 0.5; % [mm]
% lim_y = [min(Y_q(:)) - margin, max(Y_q(:)) + margin];
% lim_x = [min(X_q(:)) - margin, max(X_q(:)) + margin];

% xlim(ax_q, lim_y);
% ylim(ax_q, lim_x);

% % Explicitly set labels with units
% xlabel(ax_q, 'y [mm]', 'FontWeight', 'bold');
% ylabel(ax_q, 'x [mm]', 'FontWeight', 'bold');

% % Lock limits to prevent flickering
% set(ax_q, 'XLimMode', 'manual', 'YLimMode', 'manual');

% % Calculate Scaling Factor for Quiver
% % We want the maximum arrow to be roughly the size of a few grid cells
% max_strain_val = max(E_mod_surf(:), [], 'omitnan');
% if max_strain_val == 0, max_strain_val = 1; end

% % Target length in plot units (mm). Grid spacing is skip*dx*1000.
% grid_spacing_mm = mean(diff(Y_q(1,:))); 
% target_arrow_len_mm = 1.5 * grid_spacing_mm;
% scale_factor = target_arrow_len_mm / max_strain_val;

% fprintf('  Quiver Scale Factor: %.2e (Max Strain: %.2e)\n', scale_factor, max_strain_val);

% % Initialize Quiver Object with manual scaling (0)
% % We pass 0 as the scale argument to disable auto-scaling, and pre-multiply data by scale_factor
% q_handle = quiver(ax_q, Y_q, X_q, zeros(size(Y_q)), zeros(size(X_q)), 0, 'LineWidth', 1.5, 'Color', 'b');

% for i = 1:2:Nt 
%     Exz_curr = Exz_surf(x_roi_idx, y_roi_idx, i);
%     Eyz_curr = Eyz_surf(x_roi_idx, y_roi_idx, i);
    
%     % Update data (apply scaling manually)
%     set(q_handle, 'UData', Eyz_curr * scale_factor, 'VData', Exz_curr * scale_factor);
    
%     title(ax_q, sprintf('Shear Vector Field | t = %.2f ms', t_vec(i)*1000));
    
%     frame = getframe(fig_quiv);
%     writeVideo(v_quiver, frame);
% end
% close(v_quiver);
% close(fig_quiv);


%% 8.3 Polar Unwrapped Space-Time Plot (Optimized)
fprintf('[3/5] Generating Polar Space-Time Plots (Nature Style)...\n');

% Define Polar Grid
max_r = 20e-3; % 20mm radius
dr = grid_cfg.dx;
nr = floor(max_r / dr);
r_vec_p = (0:nr-1) * dr;
ntheta = 180;
theta_vec_p = linspace(0, 2*pi, ntheta);

[R_p, Theta_p] = ndgrid(r_vec_p, theta_vec_p);
X_p = R_p .* cos(Theta_p);
Y_p = R_p .* sin(Theta_p);

% Pre-calculate interpolation indices (since grid is constant)
[X_mesh, Y_mesh] = ndgrid(kgrid.x_vec, kgrid.y_vec);
F_interp = griddedInterpolant(X_mesh, Y_mesh, zeros(Nx, Ny), 'linear', 'none');

% Storage for polar data: [r, theta, t]
polar_data = zeros(nr, ntheta, Nt, 'single');

for i = 1:Nt
    F_interp.Values = double(E_mod_surf(:, :, i)); % Update values
    polar_data(:, :, i) = F_interp(X_p, Y_p);
end

% Create Combined Figure
fig_st = figure('Name', 'Space-Time Analysis', 'Color', 'w', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
t_layout = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel A: Radial Propagation
nexttile;
radial_map = squeeze(mean(polar_data, 2)); % [r, t]
imagesc(t_vec*1000, r_vec_p*1000, radial_map);
set(gca, 'YDir', 'normal');
colormap(gca, parula(256)); % Fallback to parula if viridis is missing, but prefer viridis
try colormap(gca, viridis(256)); catch; end
c = colorbar;
c.Label.String = 'Strain Magnitude';
ylabel('Radius r [mm]', 'FontName', 'Arial', 'FontSize', 9);
set(gca, 'FontName', 'Arial', 'FontSize', 9);

% Add Slope Annotation (Phase Velocity)
hold on;
% Assume wave travels at shear speed
v_shear = medium.sound_speed_shear; 
% Line: r = v * t + offset
% Try to align with the main wavefront. 
% For now, just show the slope characteristic.
ref_t_start = 0;
ref_r_start = 0;
ref_t_end = max(t_vec);
ref_r_end = ref_r_start + v_shear * (ref_t_end - ref_t_start);

% Draw a representative line (shifted to be visible)
% Find a point with high energy to anchor
[~, max_idx] = max(radial_map(:));
[r_max_idx, t_max_idx] = ind2sub(size(radial_map), max_idx);
t_anchor = t_vec(t_max_idx);
r_anchor = r_vec_p(r_max_idx);

% Draw line through anchor
t_line = [t_vec(1), t_vec(end)];
r_line = r_anchor + v_shear * (t_line - t_anchor);

% plot(t_line*1000, r_line*1000, 'w--', 'LineWidth', 1.5);
% text(t_line(end)*1000*0.8, r_line(end)*1000, sprintf('v_{shear} = %.1f m/s', v_shear), ...
    % 'Color', 'w', 'FontName', 'Arial', 'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
hold off;
ylim([0, max_r*1000]);

% Panel B: Angular Rotation at Fixed Radius
nexttile;
target_r = 5e-3; % 5mm
[~, r_idx] = min(abs(r_vec_p - target_r));
angular_map = squeeze(polar_data(r_idx, :, :));
imagesc(t_vec*1000, rad2deg(theta_vec_p), angular_map);
set(gca, 'YDir', 'normal');
colormap(gca, parula(256)); 
try colormap(gca, viridis(256)); catch; end
colorbar;
xlabel('Time [ms]', 'FontName', 'Arial', 'FontSize', 9);
ylabel('Angle [deg]', 'FontName', 'Arial', 'FontSize', 9);
title(sprintf('Angular Rotation (r = %.1f mm)', target_r*1000), 'FontName', 'Arial', 'FontSize', 9);
set(gca, 'FontName', 'Arial', 'FontSize', 9);

% Export
exportgraphics(fig_st, '3_SpaceTime_Analysis.svg', 'ContentType', 'vector');
close(fig_st);


%% 8.4 Phase Map (Windmill) (Optimized)
fprintf('[4/5] Generating Phase Map Snapshots (Nature Style)...\n');

% Select time points
snapshot_times = [0.25, 0.5, 0.75] * target.sim_duration;
ds_q = 8; % Quiver downsample

for i = 1:length(snapshot_times)
    [~, t_idx] = min(abs(t_vec - snapshot_times(i)));
    
    exz_snap = Exz_surf(:, :, t_idx);
    eyz_snap = Eyz_surf(:, :, t_idx);
    mod_snap = E_mod_surf(:, :, t_idx);
    
    phase_snap = atan2(eyz_snap, exz_snap); % [-pi, pi]
    
    fig_phase = figure('Color', 'w', 'Visible', 'off', 'Position', [100, 100, 600, 500]);
    
    % 1. Phase Image
    im = imagesc(kgrid.y_vec*1000, kgrid.x_vec*1000, phase_snap);
    axis image;
    colormap(hsv); 
    clim([-pi, pi]);
    
    % Transparency
    alpha_map = mod_snap / max(mod_snap(:));
    im.AlphaData = alpha_map;
    set(gca, 'Color', 'w'); % White background
    
    hold on;
    % 2. Quiver Overlay
    % Downsample
    y_q = kgrid.y_vec(1:ds_q:end) * 1000;
    x_q = kgrid.x_vec(1:ds_q:end) * 1000;
    [YQ, XQ] = meshgrid(y_q, x_q);
    
    u_q = eyz_snap(1:ds_q:end, 1:ds_q:end);
    v_q = exz_snap(1:ds_q:end, 1:ds_q:end);
    
    % Filter small arrows
    m_q = sqrt(u_q.^2 + v_q.^2);
    mask_q = m_q > max(m_q(:))*0.1;
    
    quiver(YQ(mask_q), XQ(mask_q), u_q(mask_q), v_q(mask_q), ...
        'Color', 'k', 'LineWidth', 0.8, 'MaxHeadSize', 0.5, 'AutoScaleFactor', 0.8);
    
    hold off;
    
    % Formatting
    xlabel('y [mm]', 'FontName', 'Arial', 'FontSize', 9);
    ylabel('x [mm]', 'FontName', 'Arial', 'FontSize', 9);
    set(gca, 'FontName', 'Arial', 'FontSize', 9);
    box off; % Only axes
    
    exportgraphics(fig_phase, sprintf('4_Phase_Map_t%.2fms.svg', t_vec(t_idx)*1000), 'ContentType', 'vector');
    close(fig_phase);
end


%% 8.6 Video Generation
fprintf('Generating 3D Shear Strain Visualization Video...\n');

% % Only generate Shear Strain video as requested (removing Displacement video)
% generate_visualization_video('6_Skin_Shear_Strain_3D.avi', ...
%     u_z(:, :, 1, :), ...          % Deformation data
%     epsilon_shear_mod(:, :, 1, :), ... % Color data
%     kgrid, t_vec, grid_cfg, ...
%     60, ...                       % FPS
%     2, ...                        % Scale Exaggeration
%     2, ...                        % Grid Skip
%     'Shear Strain |\epsilon_{shear}|', ... 
%     '3D Skin Shear Strain Visualization', ... 
%     [], ...                       
%     jet(256));                    

fprintf('All visualizations completed.\n');

function generate_visualization_video(video_filename, u_z_surf_4d, c_data_surf_4d, ...
                                      kgrid, t_vec, grid_cfg, ...
                                      video_fps, scale_exaggeration, grid_skip, ...
                                      c_label, title_prefix, c_limits_override, cmap_in)
    
    % --- Video Settings ---
    v = VideoWriter(video_filename, 'Motion JPEG AVI'); % Specify profile for compatibility
    v.FrameRate = video_fps;
    open(v);

    % Prepare Data
    Nt = length(t_vec);
    
    % Extract Surface Data & Downsample
    X_plot = kgrid.x_vec(1:grid_skip:end) * 1000; % [mm]
    Y_plot = kgrid.y_vec(1:grid_skip:end) * 1000; % [mm]
    [Y_mesh, X_mesh] = meshgrid(Y_plot, X_plot);
    
    % Determine global color limits
    if ~isempty(c_limits_override)
        c_limits = c_limits_override;
    else
        % Auto-detect limits
        max_val = max(c_data_surf_4d(:), [], 'omitnan');
        min_val = min(c_data_surf_4d(:), [], 'omitnan');
        
        if min_val < 0 && max_val > 0
            % Signed data: Use symmetric limits
            abs_max = max(abs([min_val, max_val]));
            c_limits = [-abs_max, abs_max];
        else
            % Positive data (Magnitude)
            c_limits = [0, max_val];
            if ~isfinite(c_limits(2)) || c_limits(2) <= c_limits(1)
                c_limits = [0 1];
            end
        end
    end

    % Determine Z-Scaling based on DEFORMATION data (u_z_surf_4d)
    uz_surface_downsampled = u_z_surf_4d(1:grid_skip:end, 1:grid_skip:end, 1, :);
    limit_xy = grid_cfg.Lx/2*1000;
    z_visual_ratio = 0.12;
    zmax_mm = max(abs(uz_surface_downsampled(:)) * 1000, [], 'omitnan');
    
    if isempty(zmax_mm) || ~isfinite(zmax_mm)
        zmax_mm = 0;
    end
    
    target_z_half_mm = limit_xy * z_visual_ratio;
    if ~isfinite(target_z_half_mm) || target_z_half_mm <= 0
        target_z_half_mm = 1;
    end
    
    if zmax_mm > 0
        z_scale = scale_exaggeration * (target_z_half_mm / zmax_mm);
    else
        z_scale = scale_exaggeration;
    end
    z_limits = [-1 1] * (target_z_half_mm * 1.05);

    % Setup Figure
    fig_vid = figure('Name', title_prefix, ...
                     'Color', 'w', 'Units', 'pixels', 'Position', [100, 100, 800, 600], 'Visible', 'on');
    ax_vid = axes('Parent', fig_vid);

    % Loop through time steps
    frame_indices = 1:10:Nt;
    if isempty(frame_indices)
        frame_indices = 1;
    end

    % Initialize Plot with first frame
    i0 = frame_indices(1);
    
    % Data for frame i0
    uz_curr = squeeze(u_z_surf_4d(1:grid_skip:end, 1:grid_skip:end, 1, i0)); 
    c_curr  = squeeze(c_data_surf_4d(1:grid_skip:end, 1:grid_skip:end, 1, i0));
    
    uz_curr(isnan(uz_curr)) = 0;
    c_curr(isnan(c_curr)) = 0;
    Z_deform = uz_curr * 1000 * z_scale;

    h = surf(ax_vid, Y_mesh, X_mesh, Z_deform, c_curr);
    set(h, 'EdgeColor', 'none', ...
           'FaceColor', 'interp', ...
           'FaceLighting', 'gouraud', ...
           'AmbientStrength', 0.6, ...
           'DiffuseStrength', 0.8, ...
           'SpecularStrength', 0.2, ...
           'SpecularExponent', 10);

    colormap(ax_vid, cmap_in);
    clim(ax_vid, c_limits);
    cb = colorbar(ax_vid);
    cb.Label.String = c_label;
    cb.Label.FontSize = 10;

    light(ax_vid, 'Position',[-50 -50 100],'Style','local', 'Color', [1 1 1]);
    light(ax_vid, 'Position',[ 50  50 50], 'Style','local', 'Color', [0.8 0.8 0.8]);

    axis(ax_vid, 'equal');
    axis(ax_vid, [-limit_xy limit_xy -limit_xy limit_xy z_limits(1) z_limits(2)]);
    grid(ax_vid, 'on');
    set(ax_vid, 'GridAlpha', 0.1);

    xlabel(ax_vid, 'y [mm]');
    ylabel(ax_vid, 'x [mm]');
    zlabel(ax_vid, 'Exaggerated u_z [mm]');
    view(ax_vid, 45, 30);

    title(ax_vid, sprintf('%s | t = %.2f ms', title_prefix, t_vec(i0)*1000), ...
          'FontSize', 14, 'FontWeight', 'bold');

    drawnow;
    
    % Video Loop
    for i = frame_indices
        
        uz_curr = squeeze(u_z_surf_4d(1:grid_skip:end, 1:grid_skip:end, 1, i)); 
        c_curr  = squeeze(c_data_surf_4d(1:grid_skip:end, 1:grid_skip:end, 1, i));
        
        uz_curr(isnan(uz_curr)) = 0;
        c_curr(isnan(c_curr)) = 0;
        Z_deform = uz_curr * 1000 * z_scale;
        
        set(h, 'ZData', Z_deform, 'CData', c_curr);

        title(ax_vid, sprintf('%s | t = %.2f ms', title_prefix, t_vec(i)*1000), ...
              'FontSize', 14, 'FontWeight', 'bold');
        
        drawnow;
        
        frame = getframe(fig_vid);
        writeVideo(v, frame);
    end

    close(v);
    close(fig_vid);
    fprintf('Video saved to %s\n', video_filename);
end
