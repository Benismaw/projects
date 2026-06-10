//classe pour les champs (Fields)
package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.ClassDefinition;


public abstract class AbstractDeclField extends Tree{

    // Méthodes de vérification à implémenter en Passe 2 et 3
    protected abstract void verifyFieldMembers(DecacCompiler compiler, 
            ClassDefinition currentClass, int index) throws ContextualError;

    protected abstract void verifyFieldBody(DecacCompiler compiler, 
            ClassDefinition currentClass) throws ContextualError;

    protected abstract void codeGenDefaultInit(DecacCompiler compiler);

    protected abstract void codeGenExplicitInit(DecacCompiler compiler);

}