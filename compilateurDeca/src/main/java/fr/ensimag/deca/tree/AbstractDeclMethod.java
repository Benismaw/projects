//classe pour les methodes 
package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.MethodDefinition;


public abstract class AbstractDeclMethod extends Tree{

    // Méthodes de vérification à implémenter en Passe 2 et 3
    protected abstract void verifyMethodMembers(DecacCompiler compiler,ClassDefinition superClass,
            ClassDefinition currentClass) throws ContextualError;

    protected abstract void verifyMethodBody(DecacCompiler compiler,
            ClassDefinition currentClass) throws ContextualError;

    public abstract AbstractIdentifier getNomMethod();
    public abstract MethodDefinition getMethodDefinition();

    public abstract void codeGenDeclMethod(DecacCompiler compiler, String className);
}