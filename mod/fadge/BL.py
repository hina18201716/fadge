def BoyerLindquist(aspin=0.0, ndim = 4 ,**kwargs):
    
    assert ndim == 4
    aa = aspin * aspin

    @jit
    def metric(x): 
        
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        
        rr = r * r 
        lamda = 0.00  # deviation parameter 
        
        P = rr + aa * np.cos(theta) * np.cos(theta) 
        Q = rr + aa - 2 * (1 + lamda * lamda ) * r 

        g = np.zeros((4, 4), **kwargs)

        
        g[0][0] =  -1 + 2*r/P           
        g[1][1] = P/Q
        g[2][2] = P
        g[3][3] = (rr + aa + 2 * aa * r * np.sin(theta) * np.sin(theta) / P ) * np.sin(theta) ** 2
        
        g[0][3] = - 4 * aspin * r * np.sin(theta) * np.sin(theta) / P
        g[3][0] = g[0][3]
        
        return g 
        
    # # BL to Cartesian KS
    # h = np.sqrt( 1 - aa )
    # L = np.ln( r)
    # x[0] = s[0] + 
    
    return metric