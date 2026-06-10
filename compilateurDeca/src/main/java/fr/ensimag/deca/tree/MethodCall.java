package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.ExpDefinition;
import fr.ensimag.deca.context.MethodDefinition;
import fr.ensimag.deca.context.Signature;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.ADDSP;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.BSR;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.STORE;
import fr.ensimag.ima.pseudocode.instructions.SUBSP;

public class MethodCall extends AbstractExpr {
    private AbstractExpr expr;
    private AbstractIdentifier methodName;
    private ListExpr arguments;

    public MethodCall(AbstractExpr expr, AbstractIdentifier methodName, ListExpr arguments) {
        this.expr = expr;
        this.methodName = methodName;
        this.arguments = arguments;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {

        // Vérifier l'expression à gauche (l'objet)
        Type typeExpr = expr.verifyExpr(compiler, localEnv, currentClass);

        // On ne peut appeler une méthode que sur une classe
        if (!typeExpr.isClass()) {
            throw new ContextualError("Appel de méthode impossible sur le type " + typeExpr, getLocation());
        }

        // Récupérer la définition de la classe dans l'environnement global
        ClassDefinition classDef = (ClassDefinition) compiler.environmentType.defOfType(typeExpr.getName());

        // Chercher la méthode dans les membres de cette classe
        ExpDefinition def = classDef.getMembers().get(methodName.getName());
        if (def == null || !def.isMethod()) {
            throw new ContextualError("La méthode '" + methodName.getName() + "' n'est pas définie dans la classe " + typeExpr.getName(), getLocation());
        }
        MethodDefinition methodDef = (MethodDefinition) def;

        // Vérifier la signature (nombre d'arguments)
        Signature sig = methodDef.getSignature();
        if (sig.size() != arguments.size()) {
            throw new ContextualError("Nombre d'arguments incorrect pour la méthode '" + methodName.getName()
                    + "'. Attendu: " + sig.size() + ", Trouvé: " + arguments.size(), getLocation());
        }

        // Vérifier le type de chaque argument par rapport à la signature
        int i = 0;
        for (AbstractExpr arg : arguments.getList()) {
            Type argType = arg.verifyExpr(compiler, localEnv, currentClass);
            Type paramType = sig.paramNumber(i);

            // Vérification de compatibilité
            if (!Type.assignCompatible(compiler.environmentType, paramType, argType)) {
                throw new ContextualError("Type d'argument incompatible à la position " + (i + 1)
                        + ". Attendu: " + paramType + ", Trouvé: " + argType, getLocation());
            }

            // Si la méthode veut un float mais reçoit un int
            if (paramType.isFloat() && argType.isInt()) {
                // On crée un noeud de conversion
                ConvFloat conv = new ConvFloat(arg);
                // On vérifie ce nouveau noeud
                conv.verifyExpr(compiler, localEnv, currentClass);
                // On remplace l'ancien argument par la conversion dans la liste
                arguments.set(i, conv);
            }
            i++;
        }

        // Décoration et typage du nœud MethodCall
        methodName.setDefinition(methodDef);
        setType(methodDef.getType());
        return methodDef.getType();
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {

        // On empile les arguments
        int nbArgs = arguments.getList().size();
        compiler.addInstruction(new ADDSP(nbArgs + 1));

        expr.codeGenExpr(compiler, register);
        compiler.addInstruction(new STORE(register, new RegisterOffset(0, Register.SP)));

        int index = -1;
        for (AbstractExpr arg : arguments.getList()) {
            arg.codeGenExpr(compiler, register);
            compiler.addInstruction(new STORE(register, new RegisterOffset(index --, Register.SP)));
        }

        // Vérification de nullité
        compiler.addInstruction(new LOAD(new RegisterOffset(0, Register.SP), register));
        compiler.addInstruction(new CMP(new NullOperand(), register));
        compiler.addInstruction(new BEQ(new Label("dereferencement_null")));

        // On charge l'adresse de la VTable dans R0
        compiler.addInstruction(new LOAD(new RegisterOffset(0, register), Register.R0));
        //Label methodLabel = methodName.getMethodDefinition().getLabel();
        int methodIndex = methodName.getMethodDefinition().getIndex();
        compiler.addInstruction(new BSR(new RegisterOffset(methodIndex, Register.R0)));

        // Nettoyage de la pile
        // On retire les arguments + l'objet 'this' (donc size + 1)
        int argsSize = arguments.getList().size();
        compiler.addInstruction(new SUBSP(argsSize + 1));

        // Récupération du résultat
        if (!register.equals(Register.R0)) {
            compiler.addInstruction(new LOAD(Register.R0, register));
        }
    }

    @Override
    public void decompile(IndentPrintStream s) {
        expr.decompile(s);
        s.print(".");
        methodName.decompile(s);
        s.print("(");
        arguments.decompile(s);
        s.print(")");
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        expr.iter(f);
        methodName.iter(f);
        arguments.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        expr.prettyPrint(s, prefix, false);
        methodName.prettyPrint(s, prefix, false);
        arguments.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenInstARM'");
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenExprARM'");
    }
}