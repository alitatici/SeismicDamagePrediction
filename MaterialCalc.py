from Units import *
import numpy as np


class Loads:
    def __init__(self, span_length, story_height, beam_width, beam_depth, column_width, column_depth, num_span,
                 num_story):
        self.span_length = span_length
        self.story_height = story_height
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.column_width = column_width
        self.column_depth = column_depth
        self.num_span = num_span
        self.num_story = num_story
        self.beam_length = self.__calc_beam_length()
        self.column_height = self.__calc_column_height()

    def __calc_beam_length(self):
        if isinstance(self.span_length, float) == 1:
            beam_length = np.repeat(np.array(self.span_length), self.num_span)
        else:
            beam_length = np.array(self.span_length)

        return beam_length

    def __calc_column_height(self):
        # if isinstance(self.story_height, float) == 1:
        column_height = np.array(self.story_height)
        # else:
        #     column_height = np.array(self.story_height)

        return column_height

    def calc_slab_weight(self):
        # Slab weight per meter
        slab_thickness = 0.15 * m
        slab_weight_pm = self.beam_length * 2 * slab_thickness * concrete_density / 4
        return slab_weight_pm

    def calc_wall_weight(self):
        # Wall weight per meter
        wall_weight_pm = wall_density * self.column_height[1:]
        wall_weight_pm = np.append(wall_weight_pm, wall_weight_pm[0] * 0.75)
        return wall_weight_pm

    def calc_sdl_weight(self, SDL):
        # Superimposed dead load per meter
        # SDL = 2 * kN / (m ** 2)
        SDL_pm = SDL * 2 * self.beam_length / 4
        return SDL_pm

    def calc_ll_weight(self, LL):
        # Live load per meter
        # LL = 2 * kN / (m ** 2)
        LL_pm = LL * 2 * self.beam_length / 4
        return LL_pm

    def calc_beam_weight(self):
        # Beam weight per meter
        beam_weight_pm = np.array(concrete_density * self.beam_width * self.beam_depth)
        return beam_weight_pm

    def calc_beam_seismic_weight(self, SDL, LL):
        slab_weight_pm = self.calc_slab_weight()
        wall_weight_pm = self.calc_wall_weight()
        SDL_pm = self.calc_sdl_weight(SDL)
        LL_pm = self.calc_ll_weight(LL)
        beam_weight_pm = self.calc_beam_weight()
        # Total weight should be assigned to each beam is:
        total_weight = slab_weight_pm + wall_weight_pm.reshape(-1, 1) + SDL_pm + 0.3 * LL_pm + beam_weight_pm

        return total_weight.reshape(-1)

    def calc_column_seismic_weight(self):
        # Column weight per meter
        column_weight_pm = concrete_density * self.column_width * self.column_depth

        return column_weight_pm
