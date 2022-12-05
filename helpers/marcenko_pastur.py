import numpy as np


class MarcenkoPastur:
    def __init__(self, z_grid, c_):
        self.z_grid = z_grid
        self.c_ = c_

    @staticmethod
    def ce_and_cl_from_paper(c_, z_, bstar, missing_bstar, for_testing=None):
        """
        Here, missing bstar is important: if the model is incomplete,
        we should adjust the errors
        Parameters
        ----------
        for_testing : these are signals to test if random matrix theory is working
        c_ : model complexity
        z_ :
        bstar :
        missing_bstar :

        Returns
        -------

        """
        m = MarcenkoPastur.marcenko_pastur(c_, z_)
        m_prime = MarcenkoPastur.derivative_of_marcenko_pastur(m, c_, z_)
        xi = MarcenkoPastur.xi_function(c_, z_, m)
        xi_prime = MarcenkoPastur.derivative_of_xi_function(c_, z_, m_prime, xi, m)
        nu = MarcenkoPastur.nu_function(xi, z_, c_)
        nu_prime = MarcenkoPastur.derivative_of_nu_function(c_, z_, xi, xi_prime)
        nu_hat = nu + z_ * nu_prime
        ce_from_paper = bstar * nu
        cl_from_paper = bstar * nu_hat - c_ * (1 + missing_bstar) * nu_prime
        if for_testing is not None:
            print(f'----------------------------------\n'
                  f'GET READY TO SEE THE POWER OF RANDOM MATRIX THEORY\n'
                  f'-----------------------------------')
            signals_covariance = np.matmul(for_testing.times, for_testing) / for_testing.shape[0]
            eigenvalues, eigenvectors = np.linalg.eigh(signals_covariance)

            tmp = (1 / (eigenvalues + z_)).mean()
            print(f'marcenko-pastur implies that {tmp} = {m}\n and that {xi} = {c_ * tmp}')

            tmp = (tmp ** 2).mean()
            print(
                f'marcenko-pastur also implies that for derivatives {tmp} = {m_prime}\n and that {xi_prime} = {- c_ * m_prime}')

            tmp = (eigenvalues / (eigenvalues + z_)).mean()
            print(f'and it also implies that for nu {tmp} = {nu}')

            tmp = - (eigenvalues / ((eigenvalues + z_) ** 2)).mean()
            print(f'and it also implies that  for nu_prime {tmp} = {nu_prime}')

            tmp = ((eigenvalues ** 2) / ((eigenvalues + z_) ** 2)).mean()
            print(f'and it also implies that for nu_hat {tmp} = {nu_hat}')

        return ce_from_paper, cl_from_paper

    @staticmethod
    def marcenko_pastur(c_, z_):
        """
        This function computes the value of Marcenko-Pastur Theorem: m(-z, c)
        please ignore sigma_squared for now
        :param c: c = P / times
        :param z: a vector of ridge penalty parameter
        :return: value of m(-z)
        """
        sqrt_term = np.sqrt(((1 - c_) + z_) ** 2 + 4 * c_ * z_)
        tmp = - ((1 - c_) + z_) + sqrt_term
        tmp = tmp / (2 * c_ * z_)
        return tmp

    @staticmethod
    def derivative_of_marcenko_pastur(m, c, z):
        """
        This function computes the derivative of Marcenko-Pastur Theorem: m'(-z, c)
        please ignore sigma_squared for now
        :param sigma_squared: set as 1
        :param c: c = M/(N*times)
        :param z: a vector of ridge penalty parameter
        :return: the value of m'(-z, c)
        """
        numerator = c * (m ** 2) + m
        denominator = 2 * c * z * m + ((1 - c) + z)
        tmp = numerator / denominator
        return tmp

    @staticmethod
    def xi_function(c, z, m):  # Only works when Sigma is an identity matrix
        """
        This function computes xi(z) by Eq. (18) in Prop 15
        :param c: c = M/(N*times)
        :param z: a vector of ridge penalty parameter
        :param m: m = marcenko_pasturxi_function(1, c, z)
        :return: xi = (1 - z * m) / ((1 / c) - 1 + z * m)
        """
        return (1 - z * m) / ((1 / c) - 1 + z * m)

    @staticmethod
    def derivative_of_xi_function(c, z, derivative_of_m, xi, m):
        """
        This function computes xi'(z) by Lemma 16
        :param c: c = M/(N*times)
        :param z: z = a vector of ridge penalty parameter
        :param m: m = marcenko_pastur(1, c, z)
        :param derivative_of_m:
        :param xi: xi = xi_function(c, z, m)
        :return: xi'(z)
        """

        return c * (z * derivative_of_m - m) * (1 + xi) ** 2

    @staticmethod
    def nu_function(xi, z, c):
        '''
        This function computes \nu(z) according to Eq. (32) in Prop 7
        :param xi:
        :param z:
        :param c:
        :param Psi:
        :param M:
        :return:
        '''

        nu = 1 - z * xi / c
        return nu

    @staticmethod
    def derivative_of_nu_function(c, z, xi, xi_prime):
        '''
        This function computes the first derivative of \nu(z) according to Eq. (38)
        :param c:
        :param z:
        :param xi:
        :param xi_prime:
        :return:
        '''
        nu_prime = - (1 / c) * (xi + z * xi_prime)
        return nu_prime

    @staticmethod
    def solve_for_mcz_in_sdf(psi_eig: np.ndarray,
                             z_: np.ndarray,
                             c_: float,
                             tolerance: float = 1e-10,
                             maxit: int = 100):
        """
        Compute the function m(-z; c) in the pricing kernel paper according to equation 20

        According to eqn 20, m(-z; c) = P^{-1} \sum_i (lam_psi_i (1 - c + cz m(- z; c)) + z)^{-1}. The derivative of the
        right hand side is f'(m) = -P^{-1} \sum_i (lam_psi_i c z)/((lam_psi_i (1 - c + cz m(-z; c)) + z)^2)
        and we will use these explicit expressions in Newton's method

        To get the starting point, note that when c is small (close to 0), m(z;c) = P^{-1} \sum_i (\lam_psi_i + z)^{-1}.
        We will use this as our starting point

        :param psi_eig:
        :param z_:
        :param c_:
        :param tolerance:
        :param maxit:
        :return:
        :rtype:
        """
        psi_eig = psi_eig.reshape(-1, 1)

        starting_point = (1 / (psi_eig + z_)).mean(axis=0)  # start at m(z;c) = P^{-1} \sum_i (\lam_psi_i + z)^{-1}

        solution = starting_point.copy()

        iter = 0
        error = np.array(10 ** 10)
        while iter < maxit and np.max(np.abs(error)) > tolerance:
            # error = f(m) - m
            error = ((1 / ((psi_eig * (1 - c_ + c_ * z_ * solution)) + z_)).mean(axis=0) - solution)
            # error slope = f'(m) - 1
            error_slope = (- ((psi_eig * c_ * z_) / ((psi_eig * (1 - c_ + c_ * z_ * solution)) + z_) ** 2).mean(
                axis=0) - 1)
            solution -= error / error_slope
            iter += 1

        if not (solution > 0).all():
            # for some z we get the negative root!
            # For those z, we solve for (f(m) - m) / (m - m') = 0, where m' are the negative roots
            bad_zs = z_[solution < 0]
            # the initial guess is critical as the function is very convex. We set it to be close to the
            # point where m is not defined (close to the asymptotic line). Otherwise the solve will break
            solution_new = 1 / (c_ * bad_zs) * (c_ - 1 - bad_zs / psi_eig[0]) + 1e-5  # set a large new initial guess
            neg_solutions = solution[solution < 0].copy()
            iter = 0
            error = np.array(10 ** 10)
            while iter < maxit and np.max(np.abs(error)) > tolerance:
                # error = f(m) - m
                error = ((1 / ((psi_eig * (1 - c_ + c_ * bad_zs * solution_new)) + bad_zs)).mean(axis=0) - solution_new) \
                        / (solution_new - neg_solutions)
                # error slope = f'(m) - 1
                deriv_numerator = - ((psi_eig * c_ * bad_zs) /
                                     ((psi_eig * (1 - c_ + c_ * bad_zs * solution_new)) + bad_zs) ** 2).mean(axis=0) - 1
                error_slope = (deriv_numerator - error) / (solution_new - neg_solutions)
                solution_new -= error / error_slope
                iter += 1

            solution[solution < 0] = solution_new

        return solution
