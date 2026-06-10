package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.tools.IndentPrintStream;



public class ListDeclMethod extends TreeList<AbstractDeclMethod> {

    @Override
    public void decompile(IndentPrintStream s) {
        for (AbstractDeclMethod c : getList()) {
            c.decompile(s);
            s.println();
        }
    }
    //passe 2
    public void verifyListMethod(DecacCompiler compiler, ClassDefinition superClass, 
        ClassDefinition currentClass) throws ContextualError {
             for (AbstractDeclMethod m : getList()) {
                m.verifyMethodMembers(compiler,superClass,currentClass);

            }
        }
    // //passe 3
    public void verifyListMethodBody(DecacCompiler compiler, 
            ClassDefinition currentClass) throws ContextualError {
        for (AbstractDeclMethod m : getList()) {
            m.verifyMethodBody(compiler, currentClass);
        }
    }
}
