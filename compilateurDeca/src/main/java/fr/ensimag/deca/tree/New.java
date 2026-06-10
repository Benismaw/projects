package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.BSR;
import fr.ensimag.ima.pseudocode.instructions.LEA;
import fr.ensimag.ima.pseudocode.instructions.NEW;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;
import fr.ensimag.ima.pseudocode.instructions.STORE;

public class New extends AbstractExpr {
    private AbstractIdentifier type;

    public New(AbstractIdentifier type) {
        this.type = type;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {
        // règle 3.42 : le type doit être une classe
        Type t = type.verifyType(compiler);
        if (!t.isClass()) {
            throw new ContextualError("Le type instancié par new doit être une classe. Trouvé: " + t + " (regle : 3.42)", getLocation());
        }
        setType(t);
        return t;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        // ClassDefinition classDef= (ClassDefinition) compiler.environmentType.get(type.getName());
        ClassDefinition classDef = (ClassDefinition) type.getDefinition();
        //Calcul de la taille :d = nombre de champs + 1 (vtable)
        int taille =classDef.getNumberOfFields()+1;
        compiler.addInstruction(new NEW(taille,register));

        //verifier si le tas est plein
        compiler.addInstruction(new BOV(new Label("tas_plein")));

        // Initialiser  du pointeur vers la vtable
        DAddr vtableAddr=classDef.getVtableAddr();
        compiler.addInstruction(new LEA(vtableAddr,Register.R0));
        compiler.addInstruction(new STORE(Register.R0,new RegisterOffset(0, register)));

        //appel du sous prog d'initialisation
        compiler.addInstruction(new PUSH(register));//On empile l'adresse de l'objet comme paramètre de init.A
        compiler.addInstruction(new BSR(new Label("init." + type.getName().getName())));
        // on depile apres l'appel
        compiler.addInstruction(new POP(register));
        //compiler.addInstruction(new SUBSP(1));
    }   

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("new ");
        type.decompile(s);
        s.print("()");
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        type.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        type.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenExprARM'");
    }
}