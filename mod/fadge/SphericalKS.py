def SpericalKS(aspin=0.0, ndim=4, **kwargs):

    assert ndim == 4
    aa  = aspin * aspin

    @jit
    def metric(x): 

        r = x[1]
        theta = x[2]
        phi = x[3]
        rr = r * r
        f = 2 * r / ( rr + aa * np.cos(theta) * np.cos(theta))
        l  = np.array([
            1.0,
            0.0,
            0.0,
            aa * np.sin(theta)**2,
        ])

        g0 = np.zeros((4, 4), **kwargs)
        g0 = g0.at[0,0].set(-1)
        g0 = g0.at[0,1].set(1)
        g0 = g0.at[2,2].set(rr + aa * np.cos(theta) * np.cos(theta) )
        g0 = g0.at[3,3].set(( rr + aa ) * np.cos(theta) * np.cos(theta) )
        g0 = g0.at[1,3].set( aspin * np.sin(theta) * np.sin(theta) )
        g0 = g0.at[3,1].set(g0[1,3])
        g0 = g0.at[1,0].set(g0[0,1])
        
        return g0 + f * l[:,np.newaxis] * l[np.newaxis,:]

        # x1 = ( r*np.cos(phi) + aspin*np.sin(phi) )* np.sin(theta)
        # x2 = ( r*np.cos(phi) - aspin*np.sin(phi) )* np.sin(theta)
        # x3 = r*np.cos(theta)

        # x[1] = x1
        # x[2] = x2
        # x[3] = x3

    return metric