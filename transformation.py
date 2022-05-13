import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


class transformation:

    '''
    Module to perform all relevant transformation, for example, pre-whitening the inputs and
    logarithm (supports log10 transformation) for the outputs.
    
    JEC 21-04-2022 turn to JAX
    
    '''

    def __init__(self, theta: jnp.ndarray, y: jnp.ndarray):
        '''
        :param: theta (np.ndarray) : matrix of size N x d

        :param: y (np.ndarray) : a vector of the output

        :param: N is the number of training points

        :param: d is the dimensionality of the problem
        '''
        # input
        self.theta = theta

        msg = 'The number of training points is smaller than the dimension of the problem. Reshape your array!'

        assert self.theta.shape[0] > self.theta.shape[1], msg

        # dimension of the problem
        self.d = self.theta.shape[1]

        # number of training points
        self.N = self.theta.shape[0]

        # y is a vector of size N
        self.y = y.reshape(self.N, 1)

        self.y_min=None
                
        
    def x_transform(self) -> jnp.ndarray:
        '''
        Transform the inputs (pre-whitening step)

        :return: theta_trans (np.ndarray) : transformed input parameters
        
        TODO (JEC - 3Mai22): use an algo that do not depend on sign-convention of SVD
               On CPU/GPU they are different implementations and LAPACK on CPU can differ using
               OpenBLas or MKL.
               
               Hint : use "eign" decomposition of cov matrix
               
        4mai22: even if the mu_matrix is not the same as we apply it to train and test in the same manner
                this is not a problem in fact
        
        '''

        # calculate the covariance of the inputs
        cov = jnp.cov(self.theta.T)

        # calculate the Singular Value Decomposition
        a, b, c = jnp.linalg.svd(cov)   ##### JEC sign convention depends on implemention (CPU/GPU...Lapack/OpenBlas...)

        # see PICO paper for this step
        m_diag = jnp.diag(1.0 / jnp.sqrt(b))

        # the transformation matrix
        self.mu_matrix = jnp.dot(m_diag, c)      ## depends on sign convention but applied coherently train/test

        # calculate the transformed input parameters
        theta_trans = jnp.dot(self.mu_matrix, self.theta.T).T

        # store the transformed inputs
        self.theta_trans = theta_trans

        return theta_trans

    def x_transform_test(self, xtest: jnp.ndarray) -> jnp.ndarray:
        '''
        Given test point(s), we transform the test point in the appropriate basis

        :param: xtext (np.ndarray) : (N*,d) JEC

        :return: x_trans (np.ndarray) : the transformed input parameters
        '''
        
        # reshape the input
        xtest = xtest if xtest.ndim > 1 else xtest[jnp.newaxis,:]
        
        x_trans = jnp.dot(self.mu_matrix,xtest.T).T

        return x_trans

    def y_transform(self) -> jnp.ndarray:
        '''
        Transform the output (depends on whether we want this criterion)

        If all the outputs are positive, then y_min = 0,
        otherwise the minimum is computed and the outputs are shifted by
        this amount before the logarithm transformation is applied

        :return: y_trans (np.ndarray) : array for the transformed output
        '''
        def funcPos(y):
            # set the minimum to 0.0
            y_min = 0.0

            # calculate the logarithm of the outputs
            y_trans = jnp.log10(y)

            return y_min, y_trans

        def funcNeg(y):
            y_min = jnp.amin(y)

            # calcualte the logarithm of the outputs
            y_trans = jnp.log10(y - 2 * y_min)

            return y_min, y_trans

        y_min, y_trans= jax.lax.cond((self.y>0.0).all(),funcPos, funcNeg, self.y)
        self.y_min   = y_min
        self.y_trans = y_trans
        
        return y_trans
        

    def y_transform_test(self, y_original: jnp.ndarray) -> jnp.ndarray:
        '''
        Given a response/output which is not in the training set, this
        function will do the forward log_10 transformation.

        :param: y_original (float or np.ndarray) : original output

        :return: y_trans_test (array) : transformed output
        '''

        y_trans_test = jnp.log10(y_original - 2 * self.y_min)

        return y_trans_test

    def y_inv_transform_test(self, y_test: jnp.ndarray) -> jnp.ndarray:
        '''
        Given a response (a prediction), this function will do
        the inverse transformation (from log_10 to the original function).

        :param: y_test (float or np.ndarray) : a test (transformed) response (output)

        :return: y_inv (np.ndarray) : original (predicted) output
        '''

        y_inv = jnp.power(10, y_test) + 2.0 * self.y_min

        return y_inv
