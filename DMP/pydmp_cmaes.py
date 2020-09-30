#########################################################
# Copyright (C) 2013 Travis DeWolf
# Copyright (C) 2019 Ronaldson Bellande and Sam Pickell
# Last Updated December 9, 2019
# pydmp_cmaes.py
#########################################################

#  NOTE the gen_weights function is based on the work by Travis DeWolf who
#  made the pydmps library we used after switching from Ronaldson's DMP code.
#  Included in this file is the work we put in to trying to modify the weight
#  calculation to use CMA_ES in place of linear regression. Due to a lack of
#  time and lack of obvious way to separate/calculate the objective function,
#  and due to the cma "cma.ff.fun_as_arg" command not accepting any objective
#  function we gave it, we were unsuccessful in getting the code to work and
#  produce results. We have left only the gen_weights function here rather than
#  the entire DMP_Discrete class and test code, as it is the only function that
#  we modified. However, while testing, all that code was here, and worked with
#  linear regression, but not CMA_ES.

from pydmps.dmp import DMPs
import cma
import numpy as np


    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.
        f_target np.array: the desired forcing term trajectory
        """

        #  linear regression - we want to change this to cma_es

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        #self.w = np.zeros((self.n_dmps, self.n_bfs))
        #for d in range(self.n_dmps):
            #spatial scaling term
            #k = (self.goal[d] - self.y0[d])
            #for b in range(self.n_bfs):
                #numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                #denom = np.sum(x_track**2 * psi_track[:, b])
                #self.w[d, b] = numer / (k * denom)
        #self.w = np.nan_to_num(self.w)

        #  CMA_ES weight calc
        for d in range(self.n_dmps):
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track**2 * psi_track[:, b])
        f_s = numer / denom

        J = 0
        for x in range(len(f_target)):
            J += ((f_target[x] - f_s)**2)

        #J = np.sum(np.exp(f_target-f_s), 2)
        black_box = cma.fmin(cma.ff.fun_as_arg(x_track),x_track,1)
        #black_box = cma.fmin(cma.ff.fun_as_arg(J),J,x_track)
        #black_box = cma.CMAEvolutionStrategy(x_track,0.1).optimize(cma.ff.linear)
        self.w = black_box[1]
