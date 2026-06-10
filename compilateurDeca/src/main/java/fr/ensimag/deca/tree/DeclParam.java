package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.ParamDefinition;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;

import java.io.PrintStream;
public class DeclParam extends AbstractDeclParam {
    private AbstractIdentifier typeParam;
    private AbstractIdentifier nomParam;


    public DeclParam(AbstractIdentifier typeParam,AbstractIdentifier nomParam){
        this.typeParam=typeParam;
        this.nomParam=nomParam;
    }



    //pour la Passe 2 : vérifier le type du paramètre et construire la signature 
    @Override
    protected  Type verifyDeclParam(DecacCompiler compiler) throws ContextualError{
            Type typeParam=this.typeParam.verifyType(compiler);
            if (typeParam.isVoid()){
                throw new ContextualError("Le type de parametre ne peut pas être Void", getLocation());
            }
            return typeParam;
    }
    
    @Override
    // pour la Passe 3 : ajouter le paramètre à l'environnement local de la méthode 
    protected void verifyParamMembers(DecacCompiler compiler, EnvironmentExp localEnv,int offset) throws ContextualError{
        // On recupere le type validé en passe 2
        Type typeParam=this.typeParam.verifyType(compiler);

        // On cree la definition du parametre
        ParamDefinition paramDef = new ParamDefinition(typeParam, getLocation());

        // On attribue l'adresse -offset(LB) à la définition
         paramDef.setOperand(new RegisterOffset(-offset, Register.LB)); // On declare l'identifiant dans l'environnement local de la methode
        try {
            localEnv.declare(this.nomParam.getName(), paramDef);
        } catch (EnvironmentExp.DoubleDefException e) {
            throw new ContextualError("Nom de paramètre dupliqué : '" + nomParam.getName() + "' (règle 3.12)", getLocation());}

        // On decore l'arbre
        this.nomParam.setDefinition(paramDef);
    }

     @Override
    public void decompile(IndentPrintStream s){
        typeParam.decompile(s);
        s.print(" ");
        nomParam.decompile(s);
    }
    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        typeParam.prettyPrint(s, prefix, false);
        nomParam.prettyPrint(s, prefix, true);
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        typeParam.iter(f);
        nomParam.iter(f);
    }
}
