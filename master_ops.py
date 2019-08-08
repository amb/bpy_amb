"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created Date: Monday, June 17th 2019, 5:39:09 pm
Copyright: Tommi Hyppänen
"""

import bpy  # noqa:F401
import numpy as np  # noqa:F401
import bmesh  # noqa:F401
from collections import OrderedDict, defaultdict  # noqa:F401
import mathutils as mu  # noqa:F401


class PanelBuilder:
    def __init__(self, master_name, input_ops, p_idname, p_spacetype, p_regiontype, p_category):
        mesh_ops = [i(master_name) for i in input_ops]

        # inject panel functionality into operators
        def _inject(cl):
            def invoke(self, context, event):
                pgroup = getattr(context.scene, self.parent_name)
                # copy property values from panel to operator
                if self.prefix != "":
                    for p in self.my_props:
                        opname = self.prefix + "_" + p
                        panel_value = getattr(pgroup, opname)
                        setattr(self, p, panel_value)
                return self.execute(context)

            def draw(self, context):
                layout = self.layout
                col = layout.column()

                for p in self.my_props:
                    row = col.row()
                    row.prop(self, p, expand=True)

            cl.draw = draw
            cl.invoke = invoke

        for m in mesh_ops:
            _inject(m.op)

        # ---
        self.panel = {
            i.prefix: bpy.props.BoolProperty(
                name=i.prefix.capitalize() + " settings",
                description="Display settings of the tool",
                default=False,
            )
            for i in mesh_ops
        }

        self.categories = set()
        self.draw_order = defaultdict(list)
        for i in mesh_ops:
            self.categories.add(i.category)
            self.draw_order[i.category].append(i)

        self.master_name = master_name
        self.mesh_ops = mesh_ops

        # The main panel
        def ptbuild(this):
            class _pt(bpy.types.Panel):
                bl_label = " ".join([i.capitalize() for i in this.master_name.split("_")])
                bl_idname = (
                    p_idname
                    + "_PT_"
                    + "".join([i.capitalize() for i in this.master_name.split("_")])
                    + "_panel"
                )

                bl_space_type = p_spacetype
                bl_region_type = p_regiontype
                bl_category = p_category

                # self here is actually the inner context, self of the built class
                def draw(self, context):
                    layout = self.layout
                    mcol = layout
                    pgroup = getattr(context.scene, this.master_name)

                    for cat in this.draw_order.keys():
                        col = mcol.box().column(align=True)
                        if len(cat) > 0:
                            col.label(text=cat)
                        for mop in this.draw_order[cat]:
                            split = col.split(factor=0.15, align=True)
                            opname = "panel_" + mop.prefix

                            if len(mop.props) == 0:
                                split.prop(pgroup, opname, text="", icon="DOT")
                            else:
                                if getattr(pgroup, opname):
                                    split.prop(pgroup, opname, text="", icon="DOWNARROW_HLT")
                                else:
                                    split.prop(pgroup, opname, text="", icon="RIGHTARROW")

                            split.operator(
                                mop.op.bl_idname, text=" ".join(mop.prefix.split("_")).capitalize()
                            )

                            if getattr(pgroup, opname):
                                box = col.column(align=True).box().column()
                                for i, p in enumerate(mop.props):
                                    # if i % 2 == 0:
                                    #     row = box.row(align=True)
                                    row = box.row(align=True)
                                    row.prop(pgroup, mop.prefix + "_" + p)

            return _pt

        self.panel_class = ptbuild(self)

    def register_params(self):
        class ConstructedPG(bpy.types.PropertyGroup):
            pass

        setattr(ConstructedPG, "__annotations__", {})

        # register operators
        for mesh_op in self.mesh_ops:
            bpy.utils.register_class(mesh_op.op)
            for k, v in mesh_op.props.items():
                ConstructedPG.__annotations__[mesh_op.prefix + "_" + k] = v
                # print("pm:", mesh_op.prefix, "_", k, "=", v)

        # register panel values
        for k, v in self.panel.items():
            ConstructedPG.__annotations__["panel_" + k] = v
            # print("pi:", "panel_", k, "=", v)

        # register panel
        bpy.utils.register_class(self.panel_class)

        # register property group
        bpy.utils.register_class(ConstructedPG)
        print("Pgroup name:", self.master_name)
        setattr(bpy.types.Scene, self.master_name, bpy.props.PointerProperty(type=ConstructedPG))

    def unregister_params(self):
        for mesh_op in self.mesh_ops:
            bpy.utils.unregister_class(mesh_op.op)
        bpy.utils.unregister_class(self.panel_class)
        delattr(bpy.types.Scene, self.master_name)


class MacroOperator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}
    my_props = []
    prefix = ""
    parent_name = ""


class OperatorGenerator:
    def generate(self):
        pass

    def init_begin(self, master_name):
        self.props = OrderedDict()
        self.parent_name = master_name

        self.payload = None
        self.prefix = ""
        self.info = ""
        self.category = ""

    def init_end(self):
        self.name = "".join(i.capitalize() for i in self.prefix.split("_"))
        self.opname = self.parent_name + "_" + self.prefix

    def create_op(self, op_type, op_prefix):
        self.op = type(
            self.name,
            (op_type,),
            {
                "bl_idname": op_prefix + "." + self.parent_name + "_" + self.prefix,
                "bl_label": " ".join(self.prefix.split("_")).capitalize(),
                "bl_description": self.info,
                "my_props": self.props.keys(),
                "prefix": self.prefix,
                "parent_name": self.parent_name,
                "payload": self.payload,
            },
        )
        setattr(self.op, "__annotations__", {})
        for k, v in self.props.items():
            self.op.__annotations__[k] = v

    def __init__(self, master_name):
        self.init_begin(master_name)
        self.generate()
        self.init_end()
        self.create_op()
