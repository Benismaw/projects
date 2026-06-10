//classe pour les champs (Fields)
package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.*;

public abstract class AbstractDeclParam extends Tree{

    //pour la Passe 2 : vérifier le type du paramètre et construire la signature 
    protected abstract Type verifyDeclParam(DecacCompiler compiler) throws ContextualError;
    
    // pour la Passe 3 : ajouter le paramètre à l'environnement local de la méthode 
    protected abstract void verifyParamMembers(DecacCompiler compiler, EnvironmentExp localEnv,int offset) throws ContextualError;
}
