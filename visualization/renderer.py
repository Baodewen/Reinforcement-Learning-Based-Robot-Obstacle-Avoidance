import math

import pygame


class Renderer:
    def __init__(self, map_size, cell_size=10, panel_width=360):
        self.map_size = map_size
        self.cell_size = cell_size
        self.map_width = map_size * cell_size
        self.map_height = map_size * cell_size
        self.panel_width = panel_width
        self.width = self.map_width + panel_width
        self.height = self.map_height
        self.screen = None
        self.title_font = None
        self.text_font = None

    def _init(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("RL Car Avoidance Demo")
            self.title_font = pygame.font.SysFont("Consolas", 22, bold=True)
            self.text_font = pygame.font.SysFont("Consolas", 16)

    def render(self, grid, car_pos, car_heading, goal_pos, lidar_points, trail, info):
        grid_height, grid_width = grid.shape
        if grid_width != self.map_size:
            self.map_size = grid_width
            self.map_width = self.map_size * self.cell_size
            self.map_height = grid_height * self.cell_size
            self.width = self.map_width + self.panel_width
            self.height = self.map_height
            if self.screen is not None:
                self.screen = pygame.display.set_mode((self.width, self.height))
        self._init()
        self.screen.fill((238, 241, 245))
        self._draw_map(grid, car_pos, car_heading, goal_pos, lidar_points, trail)
        self._draw_panel(info or {})
        pygame.display.flip()

    def _draw_map(self, grid, car_pos, car_heading, goal_pos, lidar_points, trail):
        cs = self.cell_size
        grid_height, grid_width = grid.shape
        map_rect = pygame.Rect(0, 0, self.map_width, self.map_height)
        pygame.draw.rect(self.screen, (250, 250, 252), map_rect)

        for y in range(grid_height):
            for x in range(grid_width):
                if grid[y, x] == 1:
                    pygame.draw.rect(
                        self.screen, (35, 39, 46), (x * cs, y * cs, cs, cs)
                    )

        gx, gy = goal_pos
        pygame.draw.circle(
            self.screen, (37, 161, 78), (int(gx * cs), int(gy * cs)), max(4, cs // 2)
        )

        if len(trail) > 1:
            points = [(int(px * cs), int(py * cs)) for px, py in trail[-300:]]
            pygame.draw.lines(self.screen, (44, 130, 201), False, points, 2)

        cx, cy = car_pos
        center = (int(cx * cs), int(cy * cs))
        pygame.draw.circle(self.screen, (33, 102, 172), center, max(4, cs // 2))
        head_x = cx + 0.8 * math.cos(car_heading)
        head_y = cy + 0.8 * math.sin(car_heading)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            center,
            (int(head_x * cs), int(head_y * cs)),
            2,
        )

        for point_x, point_y in lidar_points:
            pygame.draw.line(
                self.screen,
                (211, 84, 0),
                center,
                (int(point_x * cs), int(point_y * cs)),
                1,
            )

        pygame.draw.rect(self.screen, (187, 192, 200), map_rect, 1)

    def _draw_panel(self, info):
        panel_x = self.map_width
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, (22, 27, 34), panel_rect)
        pygame.draw.line(
            self.screen,
            (70, 78, 89),
            (panel_x, 0),
            (panel_x, self.height),
            2,
        )

        y = 18
        y = self._draw_text("RL Car Control Panel", panel_x + 16, y, self.title_font, (244, 246, 248))
        y += 8

        controls = info.get(
            "controls",
            "Space pause/resume | R reset | N new map | [ ] speed | Q quit",
        )
        y = self._draw_wrapped_text(controls, panel_x + 16, y, self.panel_width - 32, (172, 180, 190))
        y += 8

        sections = [
            ("Model", [
                f"file: {info.get('model_name', '-')}",
                f"episode: {info.get('episode_id', '-')}",
                f"status: {info.get('status', '-')}",
                f"map mode: {info.get('map_mode', '-')}",
            ]),
            ("State", [
                f"step: {info.get('step', '-')}",
                f"reward: {info.get('reward', '-')}",
                f"distance: {info.get('distance', '-')}",
                f"min lidar: {info.get('min_lidar', '-')}",
                f"linear vel: {info.get('linear_vel', '-')}",
                f"angular vel: {info.get('angular_vel', '-')}",
            ]),
            ("Playback", [
                f"paused: {info.get('paused', '-')}",
                f"fps: {info.get('fps', '-')}",
                f"completed episodes: {info.get('completed_episodes', '-')}",
            ]),
            ("Last Episode", [
                f"result: {info.get('last_result', '-')}",
                f"steps: {info.get('last_steps', '-')}",
                f"distance: {info.get('last_distance', '-')}",
            ]),
        ]

        for title, lines in sections:
            y = self._draw_text(title, panel_x + 16, y, self.title_font, (244, 246, 248))
            y += 2
            for line in lines:
                y = self._draw_wrapped_text(line, panel_x + 16, y, self.panel_width - 32, (214, 220, 226))
            y += 10

    def _draw_text(self, text, x, y, font, color):
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, (x, y))
        return y + surface.get_height()

    def _draw_wrapped_text(self, text, x, y, width, color):
        words = str(text).split()
        if not words:
            return y + 18
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if self.text_font.size(candidate)[0] <= width:
                line = candidate
                continue
            y = self._draw_text(line, x, y, self.text_font, color)
            line = word
        return self._draw_text(line, x, y, self.text_font, color)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
