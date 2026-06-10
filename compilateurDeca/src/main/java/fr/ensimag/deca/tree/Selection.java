package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.ExpDefinition;
import fr.ensimag.deca.context.FieldDefinition;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.LOAD;

public class Selection extends AbstractLValue {
    private AbstractExpr expr;
    private AbstractIdentifier fieldName;

    public Selection(AbstractExpr expr, AbstractIdentifier fieldName) {
        this.expr = expr;
        this.fieldName = fieldName;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {

        // On vérifie la partie gauche (l'objet)
        Type typeExpr = expr.verifyExpr(compiler, localEnv, currentClass);

        // On vérifie que c'est une classe
        if (!typeExpr.isClass()) {
            throw new ContextualError("La sélection (.) n'est possible que sur des objets. Type trouvé: " + typeExpr, getLocation());
        }

        // Chercher le champ dans la classe de l'objet
        ClassDefinition classDef = (ClassDefinition) compiler.environmentType.defOfType(typeExpr.getName());
        ExpDefinition fieldDef = classDef.getMembers().get(fieldName.getName());

        if (fieldDef == null || !fieldDef.isField()) {
            throw new ContextualError("Le champ '" + fieldName.getName() + "' n'existe pas dans la classe " + typeExpr.getName(), getLocation());
        }

        FieldDefinition field = (FieldDefinition) fieldDef;

        // Vérification de la visibilité
        if (field.getVisibility() == Visibility.PROTECTED) {
            if (currentClass == null) {
                throw new ContextualError("Accès illégal à un champ protégé", getLocation());
            }
            // Il faut que la classe courante soit une sous-classe de la classe qui définit le champ
            if (!compiler.environmentType.subtype(currentClass.getType(), field.getContainingClass().getType())) {
                throw new ContextualError("Accès interdit au champ protégé (pas dans la hiérarchie)", getLocation());
            }
            // Il faut que le type de l'objet (expr) soit un sous-type de la classe courante
            if (!compiler.environmentType.subtype(typeExpr, currentClass.getType())) {
                throw new ContextualError("Accès interdit au champ protégé (l'objet n'est pas un sous-type de la classe courante)", getLocation());
            }
        }

        fieldName.setDefinition(field);
        setType(field.getType());
        return field.getType();
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        expr.codeGenExpr(compiler, register);//on evalue l'objet(partie gauche de la selection) dans le registre
        //test de deferencement de null
        if (!compiler.getCompilerOptions().getNoCheck()) {
            compiler.addInstruction(new CMP(new NullOperand(), register));
            compiler.addInstruction(new BEQ(new Label("dereferencement_null")));
        }
        int indexField=fieldName.getFieldDefinition().getIndex();//récupérer l'index du champ depuis sa définition
        compiler.addInstruction(new LOAD(new RegisterOffset(indexField, register),register));
    }

    @Override
    public void decompile(IndentPrintStream s) {
        expr.decompile(s);
        s.print(".");
        fieldName.decompile(s);
    }

    public AbstractExpr getExpr() {
        return expr;
    }

    public FieldDefinition getFieldDefinition() {
        return (FieldDefinition) fieldName.getDefinition();
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        expr.iter(f);
        fieldName.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        expr.prettyPrint(s, prefix, false);
        fieldName.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenExprARM'");
    }
}