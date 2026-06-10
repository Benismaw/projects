package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.tools.IndentPrintStream;
import org.apache.log4j.Logger;
import fr.ensimag.deca.context.ClassDefinition;
/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class ListDeclField extends TreeList<AbstractDeclField> {

    @Override
    public void decompile(IndentPrintStream s) {
        for (AbstractDeclField c : getList()) {
            c.decompile(s);
            s.println();
        }
    }

    public void verifyListField(DecacCompiler compiler, ClassDefinition superClass, 
        ClassDefinition currentClass) throws ContextualError {
            int index=superClass.getNumberOfFields();// l'index des nouveaux champs commence apres ceux de la super-classe
             for (AbstractDeclField c : getList()) {
                index++;// Chaque DeclField s'incrémentera l'index s'il est valide
                c.verifyFieldMembers(compiler,currentClass, index);

            }
            currentClass.setNumberOfFields(index);
        }

    /**
     * Passe 3 : Vérification de l'initialisation et du corps des methodes
     */
    public void verifyListFieldBody(DecacCompiler compiler, ClassDefinition currentClass) throws ContextualError {
        for (AbstractDeclField c : getList()) {
            c.verifyFieldBody(compiler, currentClass);
        }
    }
}
