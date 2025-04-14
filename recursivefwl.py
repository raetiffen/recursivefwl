"""
    Implementation of iterated FWL and recursive FWL decompositions algorithms. Outputs decomposition tree for LaTeX. 
    
    Features:
     - Two estimation methods: iterated FWL and recursive FWL decompositions
     - Estimation by recursive FWL decompositions also outputs standard errors
     - Estimation by recursive FWL decompositions offers the option to output trees for the LaTeX "forest" package, and to first project out some covariates before estimating the tree (excluding them from the tree). The output is saved by default to 'tree.txt'. Trees with 3 or fewer covariates are modeled vertically, and 4 or more horizontally.

    Setup:
     - csv, where the first row are variable names and each column are the observations of that variable. The left-most variable is handled as the dependent variable.
     - When using this to estimate and output trees, there is the option to exclude some covariates from the tree. For this feature, add a second row under the variable names where all values are 0 or 1: variables with a 0 will be included in the tree, and variables with a 1 will be projected out (using the iterated FWL algorithm) before the tree is estimated.

"""
import sympy as sp # Used only for evaluating substitutions after iterated FWL estimation

def read_data(filename):
    # Reads data from a csv and returns a dictionary, where labels are the variable names (first row of the csv) and values are lists of observations
    with open(filename, 'r') as file:
        line_counter = 0
        for line in file:
            if line_counter == 0:
                # Create a list of variable names and prepare dictionary
                vars = line.strip().split(',')
                data = {n: [] for n in vars}
            else:
                # Add this line's values to the dictionary
                values = line.strip().split(',')
                for i in range(len(values)):
                    data[vars[i]].append(float(values[i]))
            line_counter += 1
    # Convert to tuples
    for i in data:
        data[i] = tuple(data[i])
    return data, vars

def regress(dep, ind, return_resids):
    # Performs a bivariate regression of dep on ind (both lists of observations) without a constant. Returns coefficient, and residuals if return_resids == 1.
    try:
        coeff = sum(dep[i]*ind[i] for i in range(len(dep))) / sum(i**2 for i in ind)
    except:
        print("Error in bivariate regression")
        return
    if return_resids == 1:
        return coeff, tuple((dep[i] - (coeff*ind[i]) for i in range(len(dep))))
    else:
        return coeff, 0

def iterated_fwl(data, vars, stop_point=1): 
    # Regresses the first variable in vars on the other variables using the iterated FWL algorithm
    # stop_point denotes the stopping point when using iterated FWL to project out some variables before estimating a tree - this is functionally Modified Gram-Schmidt
    subscripts = {k: [sp.symbols(k)] for k in vars}
    eqns = []
    for p in vars[:stop_point-1:-1]: # For each variable p, starting with the last variable
        for j in vars[:vars.index(p)]: # For each variable j up to p
            c, data[j] = regress(data[j],data[p],1) # Regress j on p, replace data[j] with residuals from this bivariate regression
            if stop_point == 1:
                subscripts[j].append(sp.symbols(subscripts[j][-1].name+p))
                eqns.append(sp.Eq(subscripts[j][-1],subscripts[j][-2]-c*subscripts[p][-1])) # Record the equation
    if stop_point != 1:
        return 0, data
    else:
        # Run substitutions - start with the final regression, substitute second to last into it, and so on
        final_eqn = eqns[-1]
        for m in eqns[-2::-1]:
            final_eqn = final_eqn.subs(m.lhs,m.rhs)
        final_eqn = sp.solve(final_eqn,subscripts[vars[0]][0]) # Solve for the dependent variable
        coeff_dict = final_eqn[0].as_coefficients_dict() # Get coefficients
        coeffs = [coeff_dict[subscripts[p][0]] for p in vars[1:]]
        return coeffs, data

def recursive_fwl(data, vars, level, full_tree, df_add=0):
    # Performs a regression of the first variable on the other variables using recursive FWL decompositions
    coefficients = []
    rssl = []
    stderrors = []
    tree = [0]
    if len(vars) == 2: # If bivariate regression (the base case), call regress and return coefficient
        coeff, resids = regress(data[vars[0]], data[vars[1]], 0)
        coefficients.append(coeff)
    else: # If multiple regression, decompose into p regressions of p-1 covariates
        # Generate list of regressions to decompose this one into
        new_vars = [[vars[i]] + vars[1:i] + vars[i+1:] for i in range(1, len(vars))]
        for n in new_vars: # Form and run decompositions
            try:
                n_coeffs, n_resids, rss, treej = recursive_fwl(data.copy(), n.copy(), level+1, full_tree, df_add)
                s_coeff, s_resids = regress(data[vars[0]],n_resids,0)
            except:
                print("Error in regression of {vars[0]} on {vars[1:]}.")
                return
            coefficients.append(s_coeff)
            rssl.append(rss)
            if full_tree == True:
                tree.append(treej)
    # Get predicted values
    y_hat = [sum(coefficients[n] * data[vars[n+1]][i] for n in range(len(coefficients))) for i in range(len(data[vars[0]]))]
    # Get residuals
    residuals = [data[vars[0]][i] - y_hat[i] for i in range(len(y_hat))]
    # Estimate standard errors
    if full_tree == True: # If outputting the full tree, estimate standard errors for all models
        rs = sum(u**2 for u in residuals)
        sigma = (rs/(len(data[vars[0]]) - (len(vars) - 1 + df_add)))**(1/2) # Sigma is the same independent of p
        if len(vars) == 2: # If bivariate, over tssj
            tss = tss = sum(i**2 for i in data[vars[1]])
            stderrors.append(sigma/((tss)**(1/2)))
        else:
            # Make list of SEs
            for j in rssl:
                stderrors.append(sigma/((j)**(1/2)))
        # Save to tree_storage
        tree[0] = (vars, coefficients, stderrors, level)
        if level == 0: # If main regression, make sure we return standard errors
            rs = stderrors
    else: # If not outputting the full tree, the above can be simplified to only find standard errors of the main model
        if level == 0: # If main regression, always get standard errors
            sigma = (sum(u**2 for u in residuals)/(len(data[vars[0]]) - (len(vars) - 1)))**(1/2)
            if len(vars) == 2: # If bivariate, over tssj
                tss = tss = sum(i**2 for i in data[vars[1]])
                rs = [(sigma/((tss)**(1/2)))]
            else: # Otherwise, each SE is sigma/rssj
                rs = []
                for j in rssl:
                    rs.append(sigma/((j)**(1/2)))
        elif level == 1: # If first decomposition, get rss
            rs = sum(u**2 for u in residuals)
        else:
            rs = 0
    return coefficients, residuals, rs, tree

def output(vars, coeffs, stderrors):
    # Outputs variables and coefficients, also standard errors if provided
    coeffs_formatted = [f"{c:#.7g}" for c in coeffs]
    if stderrors == 0:
        print("---------------------------")
        print("{:<10} | {:>12} |".format(vars[0],"Coefficient"))
        for p in range(len(vars) - 1):
            print("{:<10} | {:>12} |".format(vars[p+1],coeffs_formatted[p]))
        print("---------------------------")
    else:
        print("-------------------------------------------")
        stderrors_formatted = [f"{s:#.7g}" for s in stderrors]
        print("{:<10} | {:>12} | {:>12}".format(vars[0],"Coefficient","Standard error"))
        for p in range(len(vars) - 1):
            print("{:<10} | {:>12} | {:>12}".format(vars[p+1],coeffs_formatted[p],stderrors_formatted[p]))
        print("-------------------------------------------")
    return

def output_tree(tree, file, level, vert):
    # Outputs tree to file, using style_tree_item() to format equations. The boolean "vert" denotes whether the tree is formatted horizontally or vertically
    if not len(tree[0][0]) == 2:
        file.write('    '*level + "[" + style_tree_item(tree[0], vert) + "\n")
        for e in tree[1:]:
            output_tree(e, file, level+1, vert)
        file.write('    '*level + "]" + "\n")
    else:
        file.write('    '*level + "[" + style_tree_item(tree[0], vert) + "]" + "\n")
    return

def style_tree_item(item, vert):
    # Formats an equation for output
    eqn = r"{\(" + vert*r"\begin{aligned}" + r"\text{" + f"{item[0][0]}" + r"}" + " " + vert*r"&" + "= "
    for n in range(len(item[0]) - 1):
        if n == 0 and item[1][0] <= 0:
            eqn += r"\underset{" + f"({item[2][n]:.5g})" + r"}{" + f"-{abs(item[1][n]):.5g}" + r"} \cdot \text{" + f"{item[0][n+1]}" + r"}"
        else:
            eqn += r"\underset{" + f"({item[2][n]:.5g})" + r"}{" + f"{abs(item[1][n]):.5g}" + r"} \cdot \text{" + f"{item[0][n+1]}" + r"}"
        if n != (len(item[0]) - 2):
            if item[1][n+1] < 0:
                eqn += " " + vert*r"\\&\quad" + "- "
            else:
                eqn += " " + vert*r"\\&\quad" + "+ "
    eqn += vert*r"\end{aligned}" + r"\)}"
    return eqn

def main():
    print("----- Recursive FWL -----")
    # Prompt for and store data
    filename = str(input("Enter the name of your file: "))
    try:
        data, vars = read_data(filename)
    except:
        print("There was a problem with opening or parsing the file.")
        return
    # Verify no missing data
    missing = 0
    length = len(data[vars[0]])
    for n in vars[1:]:
        if len(data[n]) != length:
            missing = 1
    if missing == 1 or length == 0:
        print("Missing data")
        return
    # Prompt for estimation method
    print("Estimate by Iterated FWL (I) or Recursive FWL Decompositions (R)?")
    method = ""
    while method != "I" and method != "R":
        method = str(input("(I/R): "))
    # Iterated FWL
    if method == "I":
        try:
            coeffs, new_data = iterated_fwl(data, vars)
            output(vars, coeffs, 0) # Third argument is 0, as standard errors are not a feature in the iterated FWL method
        except:
            return
    # Recursive FWL decompositions
    if method == "R":
        print("Estimate and output full decomposition tree? (tree.txt)")
        full_tree = ""
        while full_tree != "Y" and full_tree != "N":
            full_tree = str(input("(Y/N): "))
        if full_tree == "Y":
            print("Are there variables set to be excluded from the tree?")
            noinclude = ""
            while noinclude != "Y" and noinclude != "N":
                noinclude = str(input("(Y/N): "))
            if noinclude == "Y": # If there are variables to exclude
                include = []
                exclude = []
                for j in vars:
                    if data[j][0] == 0:
                        include.append(j)
                    elif data[j][0] == 1:
                        exclude.append(j)
                    else:
                        print("Error: include/exclude markers are not readable")
                        return
                    data[j] = data[j][1:]
                new_vars = include + exclude
                c, residualized_vars = iterated_fwl(data, new_vars, len(include)) # Run iterated FWL to partial out excluded variables
                coeffs, resids, stderrors, tree = recursive_fwl(residualized_vars, include, 0, True, len(exclude)) # Estimate the tree on the transformed included variables
                output(include, coeffs, stderrors)   
            else: # If all variables are to be included in the tree
                coeffs, resids, stderrors, tree = recursive_fwl(data, vars, 0, True)
                output(vars, coeffs, stderrors)
            vert = len(tree[0][0]) > 4 # More than 3 covariates, makes the tree vertical
            # Output tree to tree.txt
            with open("tree.txt","w", newline="") as file:
                file.write(r"\begin{forest}" + "\n")
                if vert:
                    file.write(r"    for tree={grow'=0, draw, rounded corners}, forked edges," + "\n")
                output_tree(tree, file, 1, vert)
                file.write(r"\end{forest}")
        else:
            coeffs, resids, stderrors, tree = recursive_fwl(data, vars, 0, False)
            output(vars, coeffs, stderrors)

if __name__ == "__main__":
    main()
