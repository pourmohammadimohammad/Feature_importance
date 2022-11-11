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
            signals_covariance = np.matmul(for_testing.T, for_testing) / for_testing.shape[0]
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
        :param c: c = P / T
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
        :param c: c = M/(N*T)
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
        :param c: c = M/(N*T)
        :param z: a vector of ridge penalty parameter
        :param m: m = marcenko_pasturxi_function(1, c, z)
        :return: xi = (1 - z * m) / ((1 / c) - 1 + z * m)
        """
        return (1 - z * m) / ((1 / c) - 1 + z * m)

    @staticmethod
    def derivative_of_xi_function(c, z, derivative_of_m, xi, m):
        """
        This function computes xi'(z) by Lemma 16
        :param c: c = M/(N*T)
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
